use core::f32;

use dyn_stack::PodStack;
use equator::assert;
use faer::{solvers::SolverCore, Col, ColMut, ColRef, Mat, MatMut, MatRef};
use reborrow::{Reborrow, ReborrowMut};
use signtensors::{inplace_sct::CutHelper, SignMatMut};

pub(crate) enum RgbVector {
    Blowup {
        width: usize,
        kmat: SignMatrix,
        c: Col<f32>,
    },
    Columns {
        width: usize,
        r: Col<f32>,
        g: Col<f32>,
        b: Col<f32>,
    },
}

impl RgbVector {
    pub(crate) fn blowup_with_capacity(capacity: usize) -> Self {
        Self::Blowup {
            width: 0,
            kmat: SignMatrix::new(3),
            c: Col::zeros(capacity),
        }
    }

    pub(crate) fn columns_with_capacity(capacity: usize) -> Self {
        Self::Columns {
            width: 0,
            r: Col::zeros(capacity),
            g: Col::zeros(capacity),
            b: Col::zeros(capacity),
        }
    }

    pub(crate) fn as_ref(&self) -> RgbVectorRef {
        match self {
            RgbVector::Blowup { width, kmat, c } => RgbVectorRef::Blowup {
                width: *width,
                kmat,
                c: c.as_ref(),
            },
            RgbVector::Columns { width, r, g, b } => RgbVectorRef::Columns {
                width: *width,
                r: r.as_ref(),
                g: g.as_ref(),
                b: b.as_ref(),
            },
        }
    }

    pub(crate) fn as_mut(&mut self) -> RgbVectorMut {
        match self {
            RgbVector::Blowup { width, kmat, c } => RgbVectorMut::Blowup {
                width: *width,
                kmat,
                c: c.as_mut(),
            },
            RgbVector::Columns { width, r, g, b } => RgbVectorMut::Columns {
                width: *width,
                r: r.as_mut(),
                g: g.as_mut(),
                b: b.as_mut(),
            },
        }
    }

    pub(crate) fn width(&self) -> usize {
        match self {
            RgbVector::Blowup {
                width,
                kmat: _,
                c: _,
            } => *width,
            RgbVector::Columns {
                width,
                r: _,
                g: _,
                b: _,
            } => *width,
        }
    }
}

pub(crate) enum RgbVectorRef<'a> {
    Blowup {
        width: usize,
        kmat: &'a SignMatrix,
        c: ColRef<'a, f32>,
    },
    Columns {
        width: usize,
        r: ColRef<'a, f32>,
        g: ColRef<'a, f32>,
        b: ColRef<'a, f32>,
    },
}

pub(crate) enum RgbVectorMut<'a> {
    Blowup {
        width: usize,
        kmat: &'a mut SignMatrix,
        c: ColMut<'a, f32>,
    },
    Columns {
        width: usize,
        r: ColMut<'a, f32>,
        g: ColMut<'a, f32>,
        b: ColMut<'a, f32>,
    },
}

impl RgbVectorMut<'_> {
    pub(crate) fn width(&self) -> usize {
        match self {
            RgbVectorMut::Blowup {
                width,
                kmat: _,
                c: _,
            } => *width,
            RgbVectorMut::Columns {
                width,
                r: _,
                g: _,
                b: _,
            } => *width,
        }
    }
}

pub(crate) fn total_error(a: &[f32], bytes: &[u8]) -> i64 {
    assert!(a.len() == bytes.len());
    let mut err: i64 = 0;
    for (a, b) in a.iter().zip(bytes.iter()) {
        let a = a.clamp(0.0, 255.0).round() as i64;
        let b = *b as i64;
        let e = a - b;
        err += e * e
    }
    err
}

fn u8_error(c: u8, x: f32) -> f32 {
    match c {
        0 => f32::max(0.0, x - 0.5),
        255 => f32::max(0.0, 254.5 - x),
        c => f32::max(0.0, f32::abs(x - c as f32) - 0.5),
    }
}

fn u8_errors<'a>(bytes: &'a [u8], mat: MatRef<'a, f32>) -> impl Iterator<Item = f32> + 'a {
    let (nrows, ncols) = mat.shape();
    (0..ncols).flat_map(move |j| {
        (0..nrows).map(move |i| {
            let ij = i + nrows * j;
            let c = bytes[ij];
            let x = mat[(i, j)];
            u8_error(c, x)
        })
    })
}

pub(crate) fn to_u8(x: f32) -> u8 {
    assert!(x.is_finite());
    let x = x.clamp(0.0, 255.0);
    x.round() as _
}

pub(crate) fn greedy_cut(
    mat: MatRef<f32>,
    rng: &mut impl rand::Rng,
    stack: PodStack,
) -> (Col<f32>, Col<f32>) {
    let (nrows, ncols) = mat.shape();
    let mut s = Col::from_fn(nrows, |_| if rng.gen() { -1.0f32 } else { 1.0 });
    let mut t = Col::from_fn(ncols, |_| if rng.gen() { -1.0f32 } else { 1.0 });
    let two_remainder = faer::scale(2.0f32) * mat.rb();
    let two_remainder_transposed = two_remainder.transpose().to_owned();
    // let mut s_ones = vec![0u64; nrows.div_ceil(64)].into_boxed_slice();
    // let mut t_ones = vec![0u64; ncols.div_ceil(64)].into_boxed_slice();
    let (bit_rows, bit_cols) = (nrows.div_ceil(64), ncols.div_ceil(64));
    let mut s_ones = vec![0u64; bit_rows].into_boxed_slice();
    let mut t_ones = vec![0u64; bit_cols].into_boxed_slice();
    {
        // let s_ones = s_ones.rb_mut().storage_mut().col_as_slice_mut(0);
        s.as_ref().iter().enumerate().for_each(|(i, si)| {
            let pos = i / 64;
            let rem = i % 64;
            let signs = &mut s_ones[pos];
            if si.is_sign_negative() {
                *signs |= 1 << rem
            }
        });
        // let t_ones = t_ones.rb_mut().storage_mut().col_as_slice_mut(0);
        t.as_ref().iter().enumerate().for_each(|(i, ti)| {
            let pos = i / 64;
            let rem = i % 64;
            let signs = &mut t_ones[pos];
            if ti.is_sign_negative() {
                *signs |= 1 << rem
            }
        });
    }
    let s_ones = signtensors::MatMut::from_col_major_slice(&mut s_ones, bit_rows, 1, bit_rows);
    let mut s_ones = SignMatMut::from_storage(s_ones, nrows);
    let t_ones = signtensors::MatMut::from_col_major_slice(&mut t_ones, bit_cols, 1, bit_cols);
    let mut t_ones = SignMatMut::from_storage(t_ones, ncols);

    let _ = improve_greedy_cut(
        two_remainder.as_ref(),
        two_remainder_transposed.as_ref(),
        // s.as_mut(),
        // t.as_mut(),
        s_ones.rb_mut(),
        t_ones.rb_mut(),
        stack,
    );

    let mut improved_signs = 0;
    let s_signs = s_ones.storage().col_as_slice(0);
    s.iter_mut()
        .zip(s_signs.iter().flat_map(|&signs| {
            (0..64).map(move |i| if signs & (1 << i) != 0 { -1.0f32 } else { 1.0 })
        }))
        .for_each(|(si, s_sign)| {
            if *si != s_sign {
                improved_signs += 1;
                *si = s_sign
            }
        });
    let t_signs = t_ones.storage().col_as_slice(0);
    t.iter_mut()
        .zip(t_signs.iter().flat_map(|&signs| {
            (0..64).map(move |i| if signs & (1 << i) != 0 { -1.0f32 } else { 1.0 })
        }))
        .for_each(|(ti, t_sign)| {
            if *ti != t_sign {
                improved_signs += 1;
                *ti = t_sign
            }
        });

    (s, t)
}

fn improve_greedy_cut(
    two_remainder: MatRef<f32>,
    two_remainder_transposed: MatRef<f32>,
    // s: ColMut<f32>,
    // t: ColMut<f32>,
    mut s_ones: SignMatMut,
    mut t_ones: SignMatMut,
    stack: PodStack,
) -> f32 {
    let mut helper: CutHelper = CutHelper::new_with_st(
        two_remainder.as_ref(),
        two_remainder_transposed.as_ref(),
        s_ones
            .rb()
            .storage()
            .col_as_slice(0)
            .iter()
            .copied()
            .collect(),
        t_ones
            .rb()
            .storage()
            .col_as_slice(0)
            .iter()
            .copied()
            .collect(),
    );
    let mut c = [0.0f32];
    let mut c = faer::col::from_slice_mut(&mut c);
    let new_c = helper.cut_mat_inplace(
        two_remainder.as_ref(),
        two_remainder_transposed.as_ref(),
        s_ones.rb_mut(),
        c.rb_mut(),
        t_ones.rb_mut(),
        core::usize::MAX,
        stack,
    );
    new_c
    // (new_c, improved_signs)
}

impl<'short> ReborrowMut<'short> for RgbVectorMut<'_> {
    type Target = RgbVectorMut<'short>;

    fn rb_mut(&'short mut self) -> Self::Target {
        match self {
            RgbVectorMut::Blowup { width, kmat, c } => RgbVectorMut::Blowup {
                width: *width,
                kmat,
                c: c.as_mut(),
            },
            RgbVectorMut::Columns { width, r, g, b } => RgbVectorMut::Columns {
                width: *width,
                r: r.as_mut(),
                g: g.as_mut(),
                b: b.as_mut(),
            },
        }
    }
}

impl<'short> Reborrow<'short> for RgbVectorMut<'_> {
    type Target = RgbVectorRef<'short>;

    fn rb(&'short self) -> Self::Target {
        match self {
            RgbVectorMut::Blowup { width, kmat, c } => RgbVectorRef::Blowup {
                width: *width,
                kmat,
                c: c.as_ref(),
            },
            RgbVectorMut::Columns { width, r, g, b } => RgbVectorRef::Columns {
                width: *width,
                r: r.rb(),
                g: g.rb(),
                b: b.rb(),
            },
        }
    }
}

pub(crate) fn regress<'a>(
    a: &RgbTensor<f32>,
    smat: &SignMatrix,
    tmat: &SignMatrix,
    // kmat: &SignMatrix,
    rgb: RgbVectorMut<'a>,
) -> f32 {
    // dbg!(rgb.width());
    let smat = smat.as_mat_ref();
    let tmat = tmat.as_mat_ref();
    assert!(all(
        smat.ncols() == tmat.ncols(),
        a.shape() == (smat.nrows(), tmat.nrows()),
    ));
    let sts = smat.transpose() * smat;
    let ttt = tmat.transpose() * tmat;
    match rgb {
        RgbVectorMut::Blowup { width, kmat, mut c } => {
            let kmat = kmat.as_mat_ref().subcols(0, width);
            let ktk = kmat.transpose() * kmat;
            let xtx = Mat::from_fn(width, width, |i, j| sts[(i, j)] * ttt[(i, j)] * ktk[(i, j)]);
            let inv = xtx.cholesky(faer::Side::Upper).unwrap().inverse();
            let xta: Col<f32> = Col::from_fn(width, |i| {
                let dots: [f32; 3] =
                    core::array::from_fn(|c| smat.col(i).transpose() * a.mat(c) * tmat.col(i));
                kmat.col(i).transpose() * faer::col::from_slice(dots.as_slice())
            });
            let coefficients = inv * &xta;
            let aw_norm = coefficients.transpose() * xta;
            // let diff = c.rb() - coefficients.as_ref();
            // let oldnorm = c.norm_l2();
            // let newnorm = coefficients.norm_l2();
            // dbg!(
            //     oldnorm,
            //     newnorm,
            //     diff.norm_l2() / f32::max(oldnorm, newnorm)
            // );
            c.copy_from(coefficients);
            aw_norm
        }
        RgbVectorMut::Columns {
            width,
            mut r,
            mut g,
            mut b,
        } => {
            let xtx = Mat::from_fn(width, width, |i, j| sts[(i, j)] * ttt[(i, j)]);
            let inv = xtx.cholesky(faer::Side::Upper).unwrap().inverse();
            let at: [Mat<f32>; 3] = core::array::from_fn(|i| a.mat(i) * tmat);
            let xta: [Col<f32>; 3] = core::array::from_fn(|i| {
                let at = &at[i];
                Col::from_fn(width, |i| smat.col(i).transpose() * at.col(i))
            });
            let r_new = &inv * &xta[0];
            let g_new = &inv * &xta[1];
            let b_new = &inv * &xta[2];
            r.copy_from(r_new);
            g.copy_from(g_new);
            b.copy_from(b_new);
            let aw_norm =
                r.transpose() * &xta[0] + g.transpose() * &xta[1] + b.transpose() * &xta[2];
            aw_norm
            // todo!()
            // let width = sts.nrows();
            // let xtx = Mat::from_fn(width, width, |i, j| sts[(i, j)] * ttt[(i, j)]);
            // let inv = xtx.cholesky(faer::Side::Upper).unwrap().inverse();
            // let xta: [Col<f32>; 3] = core::array::from_fn(|k| {
            //     let a = a.mat(k);
            //     let at = a * tmat;
            //     faer::Col::from_fn(width, |j| smat.col(j).transpose() * at.col(j))
            // });
            // todo!()
        }
    }
}

pub(crate) fn improve_signs_then_coefficients_repeatedly(
    a: &RgbTensor<f32>,
    r: &mut RgbTensor<f32>,
    smat: &mut SignMatrix,
    tmat: &mut SignMatrix,
    rgb: RgbVectorMut<'_>,
    // kmat: &mut SignMatrix,
    // mut c: ColMut<f32>,
    mut stack: PodStack,
    max_iters: usize,
) -> (usize, usize) {
    let remainder = r;
    let (nrows, ncols) = a.shape();
    let width = rgb.width();
    // dbg!(width);
    let mut total_iterations = 0;
    let mut coefficient_updates = 0;
    match rgb {
        RgbVectorMut::Blowup { width, kmat, mut c } => {
            for _ in 0..max_iters {
                let mut improved = false;
                for j in 0..width {
                    let mut c_j = c.rb().to_owned();
                    c_j[j] = 0.0;
                    let rgb_j = RgbVectorRef::Blowup {
                        width,
                        kmat,
                        c: c_j.as_ref(),
                    };
                    let r_j = a.minus(smat, tmat, rgb_j);
                    let k_j = kmat.as_mat_ref().col(j);
                    let k = &[k_j[0], k_j[1], k_j[2]];
                    let r_j = r_j.combine_colors(k);
                    let s_j = smat.as_mat_mut().col_mut(j);
                    let t_j = tmat.as_mat_mut().col_mut(j);
                    let two_remainder = faer::scale(2.0f32) * r_j.as_ref();
                    let two_remainder_transposed = two_remainder.transpose().to_owned();
                    let (bit_rows, bit_cols) = (nrows.div_ceil(64), ncols.div_ceil(64));
                    let mut s_ones = vec![0u64; bit_rows].into_boxed_slice();
                    let mut t_ones = vec![0u64; bit_cols].into_boxed_slice();
                    {
                        // let s_ones = s_ones.rb_mut().storage_mut().col_as_slice_mut(0);
                        s_j.as_ref().iter().enumerate().for_each(|(i, si)| {
                            let pos = i / 64;
                            let rem = i % 64;
                            let signs = &mut s_ones[pos];
                            if si.is_sign_negative() {
                                *signs |= 1 << rem
                            }
                        });
                        // let t_ones = t_ones.rb_mut().storage_mut().col_as_slice_mut(0);
                        t_j.as_ref().iter().enumerate().for_each(|(i, ti)| {
                            let pos = i / 64;
                            let rem = i % 64;
                            let signs = &mut t_ones[pos];
                            if ti.is_sign_negative() {
                                *signs |= 1 << rem
                            }
                        });
                    }
                    let s_ones = signtensors::MatMut::from_col_major_slice(
                        &mut s_ones,
                        bit_rows,
                        1,
                        bit_rows,
                    );
                    let mut s_ones = SignMatMut::from_storage(s_ones, nrows);
                    let t_ones = signtensors::MatMut::from_col_major_slice(
                        &mut t_ones,
                        bit_cols,
                        1,
                        bit_cols,
                    );
                    let mut t_ones = SignMatMut::from_storage(t_ones, ncols);
                    let _ = improve_greedy_cut(
                        two_remainder.as_ref(),
                        two_remainder_transposed.as_ref(),
                        // s_j.as_mut(),
                        // t_j.as_mut(),
                        s_ones.rb_mut(),
                        t_ones.rb_mut(),
                        stack.rb_mut(),
                    );
                    let mut improved_signs = 0;
                    let s_signs = s_ones.storage().col_as_slice(0);
                    s_j.iter_mut()
                        .zip(s_signs.iter().flat_map(|&signs| {
                            (0..64).map(move |i| if signs & (1 << i) != 0 { -1.0f32 } else { 1.0 })
                        }))
                        .for_each(|(si, s_sign)| {
                            if *si != s_sign {
                                improved_signs += 1;
                                *si = s_sign
                            }
                        });
                    let t_signs = t_ones.storage().col_as_slice(0);
                    t_j.iter_mut()
                        .zip(t_signs.iter().flat_map(|&signs| {
                            (0..64).map(move |i| if signs & (1 << i) != 0 { -1.0f32 } else { 1.0 })
                        }))
                        .for_each(|(ti, t_sign)| {
                            if *ti != t_sign {
                                improved_signs += 1;
                                *ti = t_sign
                            }
                        });

                    if improved_signs != 0 {
                        total_iterations += improved_signs;
                        let mut rgb = RgbVectorMut::Blowup {
                            width,
                            kmat,
                            c: c.rb_mut(),
                        };
                        // let rgb = todo!();
                        let _ = regress(a, smat, tmat, rgb.rb_mut());
                        *remainder = a.minus(smat, tmat, rgb.rb());
                        improved = true;
                    }
                }
                if improved {
                    let mut rgb = RgbVectorMut::Blowup {
                        width,
                        kmat,
                        c: c.rb_mut(),
                    };
                    let _ = regress(a, &smat, &tmat, rgb.rb_mut());
                    *remainder = a.minus(smat, tmat, rgb.rb());
                    coefficient_updates += 1;
                } else {
                    break;
                }
            }
            (total_iterations, coefficient_updates)
        }
        RgbVectorMut::Columns {
            width,
            mut r,
            mut g,
            mut b,
        } => {
            // todo!();
            for _ in 0..max_iters {
                let mut improved = false;
                for j in 0..width {
                    let k = &[r[j], g[j], b[j]];
                    let mut r_j = r.rb().to_owned();
                    r_j[j] = 0.0;
                    let mut g_j = g.rb().to_owned();
                    g_j[j] = 0.0;
                    let mut b_j = b.rb().to_owned();
                    b_j[j] = 0.0;
                    let rgb_j = RgbVectorRef::Columns {
                        width,
                        r: r_j.as_ref(),
                        g: g_j.as_ref(),
                        b: b_j.as_ref(),
                    };
                    let r_j = a.minus(smat, tmat, rgb_j);
                    let r_j = r_j.combine_colors(k);
                    let s_j = smat.as_mat_mut().col_mut(j);
                    let t_j = tmat.as_mat_mut().col_mut(j);
                    let two_remainder = faer::scale(2.0f32) * r_j.as_ref();
                    let two_remainder_transposed = two_remainder.transpose().to_owned();
                    let (bit_rows, bit_cols) = (nrows.div_ceil(64), ncols.div_ceil(64));
                    let mut s_ones = vec![0u64; bit_rows].into_boxed_slice();
                    let mut t_ones = vec![0u64; bit_cols].into_boxed_slice();
                    {
                        // let s_ones = s_ones.rb_mut().storage_mut().col_as_slice_mut(0);
                        s_j.as_ref().iter().enumerate().for_each(|(i, si)| {
                            let pos = i / 64;
                            let rem = i % 64;
                            let signs = &mut s_ones[pos];
                            if si.is_sign_negative() {
                                *signs |= 1 << rem
                            }
                        });
                        // let t_ones = t_ones.rb_mut().storage_mut().col_as_slice_mut(0);
                        t_j.as_ref().iter().enumerate().for_each(|(i, ti)| {
                            let pos = i / 64;
                            let rem = i % 64;
                            let signs = &mut t_ones[pos];
                            if ti.is_sign_negative() {
                                *signs |= 1 << rem
                            }
                        });
                    }
                    let s_ones = signtensors::MatMut::from_col_major_slice(
                        &mut s_ones,
                        bit_rows,
                        1,
                        bit_rows,
                    );
                    let mut s_ones = SignMatMut::from_storage(s_ones, nrows);
                    let t_ones = signtensors::MatMut::from_col_major_slice(
                        &mut t_ones,
                        bit_cols,
                        1,
                        bit_cols,
                    );
                    let mut t_ones = SignMatMut::from_storage(t_ones, ncols);
                    let _ = improve_greedy_cut(
                        two_remainder.as_ref(),
                        two_remainder_transposed.as_ref(),
                        // s_j.as_mut(),
                        // t_j.as_mut(),
                        s_ones.rb_mut(),
                        t_ones.rb_mut(),
                        stack.rb_mut(),
                    );
                    let mut improved_signs = 0;
                    let s_signs = s_ones.storage().col_as_slice(0);
                    s_j.iter_mut()
                        .zip(s_signs.iter().flat_map(|&signs| {
                            (0..64).map(move |i| if signs & (1 << i) != 0 { -1.0f32 } else { 1.0 })
                        }))
                        .for_each(|(si, s_sign)| {
                            if *si != s_sign {
                                improved_signs += 1;
                                *si = s_sign
                            }
                        });
                    let t_signs = t_ones.storage().col_as_slice(0);
                    t_j.iter_mut()
                        .zip(t_signs.iter().flat_map(|&signs| {
                            (0..64).map(move |i| if signs & (1 << i) != 0 { -1.0f32 } else { 1.0 })
                        }))
                        .for_each(|(ti, t_sign)| {
                            if *ti != t_sign {
                                improved_signs += 1;
                                *ti = t_sign
                            }
                        });

                    if improved_signs != 0 {
                        total_iterations += improved_signs;
                        let mut rgb = RgbVectorMut::Columns {
                            width,
                            r: r.rb_mut(),
                            g: g.rb_mut(),
                            b: b.rb_mut(),
                        };
                        // let mut rgb = RgbVectorMut::Blowup {
                        //     width,
                        //     kmat,
                        //     c: c.rb_mut(),
                        // };
                        // let rgb = todo!();
                        let _ = regress(a, smat, tmat, rgb.rb_mut());
                        *remainder = a.minus(smat, tmat, rgb.rb());
                        improved = true;
                        // todo!();
                    }
                }
                if improved {
                    // todo!();
                    // let mut rgb = RgbVectorMut::Blowup {
                    //     width,
                    //     kmat,
                    //     c: c.rb_mut(),
                    // };
                    let mut rgb = RgbVectorMut::Columns {
                        width,
                        r: r.rb_mut(),
                        g: g.rb_mut(),
                        b: b.rb_mut(),
                    };
                    let _ = regress(a, smat, tmat, rgb.rb_mut());
                    *remainder = a.minus(smat, tmat, rgb.rb());
                    coefficient_updates += 1;
                } else {
                    break;
                }
            }
            (total_iterations, coefficient_updates)
        }
    }
}

// layout: [a0, a1, a2] where each ai is column-major
#[derive(Clone)]
pub(crate) struct RgbTensor<T> {
    nrows: usize,
    ncols: usize,
    pub(crate) data: Vec<T>,
}

impl<T> RgbTensor<T> {
    pub(crate) fn new(data: Vec<T>, nrows: usize, ncols: usize) -> Self {
        assert!(data.len() == nrows * ncols * 3);
        Self { nrows, ncols, data }
    }

    pub(crate) fn convert<U>(self, f: impl Fn(T) -> U) -> RgbTensor<U> {
        let Self { nrows, ncols, data } = self;
        RgbTensor {
            nrows,
            ncols,
            data: data.into_iter().map(f).collect(),
        }
    }

    fn color(&self, c: usize) -> &[T] {
        assert!(c < 3);
        let num = self.nrows * self.ncols;
        &self.data[num * c..][..num]
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

impl RgbTensor<f32> {
    fn mat(&self, c: usize) -> MatRef<f32> {
        faer::mat::from_column_major_slice(self.color(c), self.nrows, self.ncols)
    }

    pub(crate) fn col(&self) -> ColRef<f32> {
        faer::col::from_slice(&self.data)
    }

    pub(crate) fn combine_colors(&self, &[k0, k1, k2]: &[f32; 3]) -> Mat<f32> {
        faer::scale(k0) * self.mat(0)
            + faer::scale(k1) * self.mat(1)
            + faer::scale(k2) * self.mat(2)
    }

    pub(crate) fn minus(
        &self,
        smat: &SignMatrix,
        tmat: &SignMatrix,
        rgb: RgbVectorRef<'_>,
        // kmat: &SignMatrix,
        // c: ColRef<f32>,
    ) -> Self {
        let nrows = self.nrows;
        let ncols = self.ncols;
        let mut remainder = self.clone();
        let mut r_col = faer::col::from_slice_mut(remainder.data.as_mut_slice());
        match rgb {
            RgbVectorRef::Blowup { width, kmat, c } => {
                for (((s, t), k), coe) in smat
                    .as_mat_ref()
                    .col_iter()
                    .zip(tmat.as_mat_ref().col_iter())
                    .zip(kmat.as_mat_ref().col_iter())
                    .zip(c.iter())
                {
                    for c in 0..3 {
                        for j in 0..ncols {
                            for i in 0..nrows {
                                let cji = i + nrows * j + c * nrows * ncols;
                                r_col[cji] -= coe * k[c] * t[j] * s[i]
                            }
                        }
                    }
                }
            }
            RgbVectorRef::Columns { width, r, g, b } => {
                let st_iter = smat
                    .as_mat_ref()
                    .col_iter()
                    .zip(tmat.as_mat_ref().col_iter());
                let rgb_iter = r.iter().zip(g.iter()).zip(b.iter());
                for ((s, t), ((r, g), b)) in st_iter.zip(rgb_iter) {
                    for j in 0..ncols {
                        for i in 0..nrows {
                            let rji = i + nrows * j;
                            let gji = rji + nrows * ncols;
                            let bji = gji + nrows * ncols;
                            r_col[rji] -= *r * t[j] * s[i];
                            r_col[gji] -= *g * t[j] * s[i];
                            r_col[bji] -= *b * t[j] * s[i];
                        }
                    }
                }
            }
        }
        remainder
    }
}

// impl<T> Index<(usize, usize, usize)> for RgbTensor<T> {
//     type Output = T;

//     fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
//         todo!()
//     }
// }

#[derive(Clone)]
pub(crate) struct SignMatrix {
    nrows: usize,
    width: usize,
    data: Vec<f32>,
}

impl SignMatrix {
    pub(crate) fn new(nrows: usize) -> Self {
        Self {
            nrows,
            width: 0,
            data: vec![],
        }
    }

    // fn new_with_capacity(nrows: usize, width: usize) -> Self {
    //     Self {
    //         nrows,
    //         width,
    //         data: Vec::with_capacity(nrows * width),
    //     }
    // }

    pub(crate) fn push_col(&mut self, col: &[f32]) {
        assert!(col.len() == self.nrows);
        self.data.extend_from_slice(col);
        self.width += 1
    }

    pub(crate) fn as_mat_ref(&self) -> MatRef<f32> {
        faer::mat::from_column_major_slice(self.data.as_slice(), self.nrows, self.width)
    }

    fn as_mat_mut(&mut self) -> MatMut<f32> {
        faer::mat::from_column_major_slice_mut(self.data.as_mut_slice(), self.nrows, self.width)
    }
}

pub(crate) const GAMMA: [[f32; 3]; 4] = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0],
];
