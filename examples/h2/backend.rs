use aligned_vec::avec;
use equator::assert;
use faer::{dyn_stack::PodStack, Mat};
use itertools::Itertools;
use reborrow::ReborrowMut;
use signtensors::sct::{Sct, SctMut, SctRef};

pub(crate) struct GreedyScs {
    pub sct: Sct,
    pub remainder: Mat<f32>,
}

const ALIGN: usize = 128;

impl GreedyScs {
    pub(crate) fn correct_signs(&mut self, i: usize) {
        let (nrows, ncols) = self.shape();
        let width = self.width();
        let mut s_signs = avec![0; nrows.div_ceil(64)];
        let mut s_image = avec![0.0f32; ncols];
        let mut s_image_circ_s = avec![0.0f32; ncols];

        let SctMut { mut s, c, mut t } = self.sct.as_mut();
        let mut si = s.rb_mut().split_at_col_mut(i);
        assert!(all(si.0.ncols() == i, si.1.ncols() == width - i,));
        let mut si = si.1.rb_mut().split_at_col_mut(1).0;
        assert!(si.ncols() == 1);
        let mut ti = t.rb_mut().split_at_col_mut(i);
        assert!(all(ti.0.ncols() == i, ti.1.ncols() == width - i,));
        let mut ti = ti.1.rb_mut().split_at_col_mut(1).0;
        assert!(ti.ncols() == 1);
        let ci = &mut c[i..][..1];
        let mut ci = faer::col::from_slice_mut(ci);

        s_signs.copy_from_slice(si.rb_mut().storage().col(0).try_as_slice().unwrap());

        let mut scti = SctMut {
            s: si.rb_mut(),
            c: ci.rb_mut().try_as_slice_mut().unwrap(),
            t: ti.rb_mut(),
        };
        dbg!(self.remainder.squared_norm_l2() / nrows as f32);
        self.remainder += scti.expand();
        scti.c[0] = 0.0;

        signtensors::bitmagic::matvec_bit(
            nrows,
            ncols,
            &mut s_image,
            self.remainder.as_ref(),
            &bytemuck::cast_slice(&s_signs)[..ncols.div_ceil(16)],
        );
        s_image_circ_s.as_mut_slice().copy_from_slice(&s_image);
        for (i, assi) in s_image_circ_s.iter_mut().enumerate() {
            let i_pos = i / 64;
            let i_bit = i % 64;
            if s_signs[i_pos] & (1 << i_bit) != 0 {
                *assi *= -1.0f32;
            }
        }
        loop {
            let delta =
                self.remainder.diagonal().column_vector() - faer::col::from_slice(&s_image_circ_s);
            let best_position = delta
                .iter()
                .position_max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let best_value = delta[best_position];
            let flip_bit = match best_value.partial_cmp(&0.0f32).unwrap() {
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => {
                    // break ties (best_value == 0)
                    // require in this case that s_j = 1 // (-1)^j
                    // todo!()
                    false
                }
                std::cmp::Ordering::Greater => true,
            };
            if flip_bit {
                // println!("delta[{best_position}] = {best_value}");
                let bit = best_position % 64;
                let pos = best_position / 64;
                s_signs[pos] ^= 1 << bit;
                s_image.fill(0.0f32);
                signtensors::bitmagic::matvec_bit(
                    nrows,
                    ncols,
                    &mut s_image,
                    self.remainder.as_ref(),
                    &bytemuck::cast_slice(&s_signs)[..ncols.div_ceil(16)],
                );
                s_image_circ_s.as_mut_slice().copy_from_slice(&s_image);
                for (i, assi) in s_image_circ_s.iter_mut().enumerate() {
                    let i_pos = i / 64;
                    let i_bit = i % 64;
                    if s_signs[i_pos] & (1 << i_bit) != 0 {
                        *assi *= -1.0f32;
                    }
                }
            } else {
                // dbg!();
                break;
            }
        }
        let c = s_image_circ_s.into_iter().sum::<f32>();
        // dbg!(c);
        scti.as_mut()
            .s
            .storage_mut()
            .col_mut(0)
            .try_as_slice_mut()
            .unwrap()
            .copy_from_slice(&s_signs);
        scti.as_mut()
            .t
            .storage_mut()
            .col_mut(0)
            .try_as_slice_mut()
            .unwrap()
            .copy_from_slice(&s_signs);
        scti.c[0] = c / (nrows * ncols) as f32;
        // // dbg!(self.remainder.squared_norm_l2());
        // let sct = self.sct.as_ref();
        // let (_, si) = sct.s.split_at_col(width);
        // let (_, ti) = sct.t.split_at_col(width);
        // let ci = &sct.c[width..];
        // assert!(all(si.ncols() == 1, ti.ncols() == 1, ci.len() == 1));
        // let scti = SctRef {
        //     s: si,
        //     c: ci,
        //     t: ti,
        // };
        // // dbg!(scti.expand());
        self.remainder -= scti.expand();

        dbg!(self.remainder.squared_norm_l2() / nrows as f32);
        // todo!()
    }

    pub(crate) fn extend(&mut self, rng: &mut impl rand::Rng) {
        let (nrows, ncols) = self.shape();
        let width = self.width();
        let mut s_signs = avec![0; nrows.div_ceil(64)];
        let mut s_image = avec![0.0f32; ncols];
        let mut s_image_circ_s = avec![0.0f32; ncols];

        // fill s_signs randomly
        rng.fill_bytes(bytemuck::cast_slice_mut(&mut s_signs));
        let tail = nrows % 64;
        if tail != 0 {
            let last_bytes = s_signs.last_mut().unwrap();
            let low_bits = (1 << tail) - 1;
            *last_bytes &= low_bits;
        }

        signtensors::bitmagic::matvec_bit(
            nrows,
            ncols,
            &mut s_image,
            self.remainder.as_ref(),
            &bytemuck::cast_slice(&s_signs)[..ncols.div_ceil(16)],
        );
        s_image_circ_s.as_mut_slice().copy_from_slice(&s_image);
        for (i, assi) in s_image_circ_s.iter_mut().enumerate() {
            let i_pos = i / 64;
            let i_bit = i % 64;
            if s_signs[i_pos] & (1 << i_bit) != 0 {
                *assi *= -1.0f32;
            }
        }
        loop {
            let delta =
                self.remainder.diagonal().column_vector() - faer::col::from_slice(&s_image_circ_s);
            let best_position = delta
                .iter()
                .position_max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let best_value = delta[best_position];
            let flip_bit = match best_value.partial_cmp(&0.0f32).unwrap() {
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => {
                    // break ties (best_value == 0)
                    // require in this case that s_j = 1 // (-1)^j
                    // todo!()
                    false
                }
                std::cmp::Ordering::Greater => true,
            };
            if flip_bit {
                // println!("delta[{best_position}] = {best_value}");
                let bit = best_position % 64;
                let pos = best_position / 64;
                s_signs[pos] ^= 1 << bit;
                s_image.fill(0.0f32);
                signtensors::bitmagic::matvec_bit(
                    nrows,
                    ncols,
                    &mut s_image,
                    self.remainder.as_ref(),
                    &bytemuck::cast_slice(&s_signs)[..ncols.div_ceil(16)],
                );
                s_image_circ_s.as_mut_slice().copy_from_slice(&s_image);
                for (i, assi) in s_image_circ_s.iter_mut().enumerate() {
                    let i_pos = i / 64;
                    let i_bit = i % 64;
                    if s_signs[i_pos] & (1 << i_bit) != 0 {
                        *assi *= -1.0f32;
                    }
                }
            } else {
                // dbg!();
                break;
            }
        }
        let c = s_image_circ_s.into_iter().sum::<f32>();
        // dbg!(c);
        self.sct.s.extend_from_slice(&s_signs);
        self.sct.t.extend_from_slice(&s_signs);
        self.sct.c.push(c / (nrows * ncols) as f32);
        // dbg!(self.remainder.squared_norm_l2());
        let sct = self.sct.as_ref();
        let (_, si) = sct.s.split_at_col(width);
        let (_, ti) = sct.t.split_at_col(width);
        let ci = &sct.c[width..];
        assert!(all(si.ncols() == 1, ti.ncols() == 1, ci.len() == 1));
        let scti = SctRef {
            s: si,
            c: ci,
            t: ti,
        };
        // dbg!(scti.expand());
        self.remainder -= scti.expand();
        // dbg!(self.remainder.squared_norm_l2());
        // println!("s = {:0b}", 0b11 & s_signs[0]);
        // println!("R * s = {:?}", faer::col::from_slice(&s_image));
        // todo!()
    }

    pub(crate) fn width(&self) -> usize {
        self.sct.width()
    }

    pub(crate) fn shape(&self) -> (usize, usize) {
        self.remainder.shape()
    }
}
