use core::iter;

use crate::{bitmagic, inplace_sct::CutHelperMut, Concat, SignMatMut, SignMatRef};
use equator::assert;
use faer::{
    dyn_stack::{PodStack, SizeOverflow, StackReq},
    Mat, MatRef,
};
use rand::RngCore;
use reborrow::*;
use safetensors::View;

const ALIGN: usize = 128;

#[derive(Clone, Debug)]
pub struct Sct {
    pub s: Vec<u64>,
    pub c: Vec<f32>,
    pub t: Vec<u64>,
    pub nrows: usize,
    pub ncols: usize,
}

impl Concat for SctRef<'_> {
    type Owned = Sct;

    fn concat(sct: &[Self]) -> Self::Owned {
        assert!(sct.len() > 0);

        let nrows = sct[0].nrows();
        let ncols = sct[0].ncols();
        let mut width = 0;

        for sct in sct {
            assert!(all(sct.nrows() == nrows, sct.ncols() == ncols));
            width += sct.width();
        }

        let mut ret = Sct::new(nrows, ncols, 0);

        let s_stride = nrows.div_ceil(64);
        let t_stride = ncols.div_ceil(64);

        ret.s.reserve_exact(s_stride * width);
        ret.t.reserve_exact(t_stride * width);
        ret.c.reserve_exact(width);

        for sct in sct {
            let s = sct.s.storage();
            let t = sct.t.storage();
            let c = sct.c;

            for ((s, t), c) in iter::zip(iter::zip(s.col_iter(), t.col_iter()), c.iter()) {
                ret.s.extend_from_slice(s.try_as_slice().unwrap());
                ret.t.extend_from_slice(t.try_as_slice().unwrap());
                ret.c.push(*c);
            }
        }

        ret
    }
}

impl SctRef<'_> {
    pub fn as_ref(&self) -> SctRef<'_> {
        SctRef {
            s: self.s.rb(),
            c: self.c.rb(),
            t: self.t.rb(),
        }
    }

    pub fn nrows(&self) -> usize {
        self.s.nrows()
    }
    pub fn ncols(&self) -> usize {
        self.t.nrows()
    }
    pub fn width(&self) -> usize {
        self.c.len()
    }

    pub fn expand(&self) -> Mat<f32> {
        let Self { s, c, t } = *self;
        let mut mat = Mat::zeros(self.nrows(), self.ncols());
        bitmagic::matmul::mat_tmat_f32(mat.as_mut(), s, t, c);
        mat
    }
}

impl SctMut<'_> {
    pub fn as_ref(&self) -> SctRef<'_> {
        SctRef {
            s: self.s.rb(),
            c: self.c.rb(),
            t: self.t.rb(),
        }
    }

    pub fn as_mut(&mut self) -> SctMut<'_> {
        SctMut {
            s: self.s.rb_mut(),
            c: self.c.rb_mut(),
            t: self.t.rb_mut(),
        }
    }

    pub fn nrows(&self) -> usize {
        self.s.nrows()
    }
    pub fn ncols(&self) -> usize {
        self.t.nrows()
    }
    pub fn width(&self) -> usize {
        self.c.len()
    }

    pub fn expand(&self) -> Mat<f32> {
        self.as_ref().expand()
    }
}

#[derive(Clone, Debug)]
pub struct GreedyCuts {
    pub sct: Sct,
    pub remainder_cis: Mat<f32>,
    pub remainder_trans: Mat<f32>,
}

impl GreedyCuts {
    pub fn new(mat: MatRef<'_, f32>) -> Self {
        Self::with_capacity(mat, 0)
    }

    pub fn with_capacity(mat: MatRef<'_, f32>, width_capacity: usize) -> Self {
        let (nrows, ncols) = mat.shape();
        Self {
            sct: Sct {
                s: Vec::with_capacity(nrows.div_ceil(64) * width_capacity),
                c: Vec::with_capacity(width_capacity),
                t: Vec::with_capacity(ncols.div_ceil(64) * width_capacity),
                nrows,
                ncols,
            },
            remainder_cis: mat.to_owned(),
            remainder_trans: mat.transpose().to_owned(),
        }
    }

    pub fn norm_l2(&self) -> f32 {
        self.remainder_trans.norm_l2()
    }

    pub fn nrows(&self) -> usize {
        self.sct.nrows()
    }
    pub fn ncols(&self) -> usize {
        self.sct.ncols()
    }

    pub fn width(&self) -> usize {
        self.sct.width()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    const BLOCKSIZE: usize = 32;

    pub fn extend_scratch(&self) -> Result<StackReq, SizeOverflow> {
        let (nrows, ncols) = self.shape();
        StackReq::try_all_of([
            StackReq::try_new_aligned::<u64>(nrows.div_ceil(64), ALIGN)?,
            StackReq::try_new_aligned::<u64>(ncols.div_ceil(64), ALIGN)?,
            StackReq::try_new_aligned::<u64>(nrows.div_ceil(64), ALIGN)?,
            StackReq::try_new_aligned::<u64>(ncols.div_ceil(64), ALIGN)?,
            faer::linalg::temp_mat_req::<f32>(nrows, 1)?,
            faer::linalg::temp_mat_req::<f32>(ncols, 1)?,
            StackReq::try_new::<u64>(Ord::max(nrows, ncols))?,
            faer::linalg::temp_mat_req::<f32>(Self::BLOCKSIZE, 1)?,
        ])
    }

    pub fn extend(&mut self, width: usize, rng: &mut dyn RngCore, stack: &mut PodStack) {
        let (nrows, ncols) = self.shape();
        let cur = self.width();

        let (s_signs_old, stack) = stack.make_aligned_raw::<u64>(nrows.div_ceil(64), ALIGN);
        let (t_signs_old, stack) = stack.make_aligned_raw::<u64>(ncols.div_ceil(64), ALIGN);
        let (s_signs, stack) = stack.make_aligned_raw::<u64>(nrows.div_ceil(64), ALIGN);
        let (t_signs, stack) = stack.make_aligned_raw::<u64>(ncols.div_ceil(64), ALIGN);
        let (t_image_half, stack) = faer::linalg::temp_mat_uninit::<f32>(nrows, 1, stack);
        let (s_image_half, stack) = faer::linalg::temp_mat_uninit::<f32>(ncols, 1, stack);

        let mut t_image_half = t_image_half.col_mut(0);
        let mut s_image_half = s_image_half.col_mut(0);

        t_image_half.fill(0.0);
        s_image_half.fill(0.0);

        bitmagic::matvec_bit(
            nrows,
            ncols,
            t_image_half.rb_mut().try_as_slice_mut().unwrap(),
            self.remainder_cis.as_ref(),
            &bytemuck::cast_slice(t_signs)[..ncols.div_ceil(16)],
        );
        bitmagic::matvec_bit(
            ncols,
            nrows,
            s_image_half.rb_mut().try_as_slice_mut().unwrap(),
            self.remainder_trans.as_ref(),
            &bytemuck::cast_slice(s_signs)[..nrows.div_ceil(16)],
        );

        let mut cut = CutHelperMut {
            t_signs_old,
            s_signs_old,
            t_signs,
            s_signs,
            t_image_half,
            s_image_half,
        };

        self.sct.s.resize(nrows.div_ceil(64) * (cur + width), 0);
        self.sct.t.resize(ncols.div_ceil(64) * (cur + width), 0);
        self.sct.c.resize(width + cur, 0.0);

        let SctMut { mut s, c, mut t } = self.sct.as_mut();

        let blocksize = Self::BLOCKSIZE;

        let mut iter = 0;
        while iter < width {
            let blocksize = Ord::min(width - iter, blocksize);

            let (_, mut s) = s.rb_mut().split_at_col_mut(cur + iter);
            let (_, mut t) = t.rb_mut().split_at_col_mut(cur + iter);
            let (_, c) = c.split_at_mut(cur + iter);

            for k in 0..blocksize {
                cut.cut_mat(
                    self.remainder_cis.as_ref(),
                    self.remainder_trans.as_ref(),
                    s.rb_mut().split_at_col_mut(k + 1).0,
                    faer::col::from_slice_mut(&mut c[..k + 1]),
                    t.rb_mut().split_at_col_mut(k + 1).0,
                    rng,
                    usize::MAX,
                    stack,
                );
            }

            bitmagic::matmul::mat_tmat_f32(
                self.remainder_cis.as_mut(),
                s.rb().split_at_col(blocksize).0,
                t.rb().split_at_col(blocksize).0,
                &c[..blocksize],
            );
            self.remainder_trans
                .as_mut()
                .copy_from(self.remainder_cis.transpose());

            for c in &mut c[..blocksize] {
                *c = -*c;
            }

            iter += blocksize;
        }
    }
}

impl Sct {
    pub fn new(nrows: usize, ncols: usize, width: usize) -> Self {
        Self {
            s: vec![0u64; nrows.div_ceil(64) * width],
            c: vec![0.0; width],
            t: vec![0u64; ncols.div_ceil(64) * width],
            nrows,
            ncols,
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn width(&self) -> usize {
        self.c.len()
    }

    pub fn as_ref(&self) -> SctRef {
        let Self {
            s,
            c,
            t,
            nrows,
            ncols,
            ..
        } = self;
        SctRef {
            s: SignMatRef::from_storage(
                faer::mat::from_column_major_slice(s, nrows.div_ceil(64), self.width()),
                *nrows,
            ),
            c: c.as_slice(),
            t: SignMatRef::from_storage(
                faer::mat::from_column_major_slice(t, ncols.div_ceil(64), self.width()),
                *ncols,
            ),
        }
    }

    pub fn as_mut(&mut self) -> SctMut {
        let width = self.width();
        let Self {
            s,
            c,
            t,
            nrows,
            ncols,
            ..
        } = self;
        SctMut {
            s: SignMatMut::from_storage(
                faer::mat::from_column_major_slice_mut(s, nrows.div_ceil(64), width),
                *nrows,
            ),
            c,
            t: SignMatMut::from_storage(
                faer::mat::from_column_major_slice_mut(t, ncols.div_ceil(64), width),
                *ncols,
            ),
        }
    }

    pub fn expand(&self) -> Mat<f32> {
        self.as_ref().expand()
    }
}

#[derive(Copy, Clone)]
pub struct SctRef<'a> {
    pub s: SignMatRef<'a>,
    pub c: &'a [f32],
    pub t: SignMatRef<'a>,
}

pub struct SctMut<'a> {
    pub s: SignMatMut<'a>,
    pub c: &'a mut [f32],
    pub t: SignMatMut<'a>,
}

impl Sct {
    pub fn views(&self) -> impl IntoIterator<Item = (String, impl View + '_)> {
        let Self { s, c, t, .. } = self;
        struct DynView<'a>(Box<dyn View + 'a>);
        impl View for DynView<'_> {
            fn dtype(&self) -> safetensors::Dtype {
                self.0.dtype()
            }

            fn shape(&self) -> &[usize] {
                self.0.shape()
            }

            fn data(&self) -> std::borrow::Cow<[u8]> {
                self.0.data()
            }

            fn data_len(&self) -> usize {
                self.0.data_len()
            }
        }
        struct Slice<'a, T>(&'a [T], usize);
        trait DType {
            fn dtype() -> safetensors::Dtype;
        }

        impl<'a, T> Slice<'a, T> {
            pub fn new(slice: &'a [T]) -> Self {
                Self(slice, slice.len())
            }
        }

        impl<'a> DynView<'a> {
            pub fn new(value: impl View + 'a) -> Self {
                Self(Box::new(value))
            }
        }

        impl<T: DType + bytemuck::Pod> View for Slice<'_, T> {
            fn dtype(&self) -> safetensors::Dtype {
                T::dtype()
            }

            fn shape(&self) -> &[usize] {
                std::slice::from_ref(&self.1)
            }

            fn data(&self) -> std::borrow::Cow<[u8]> {
                bytemuck::cast_slice(self.0).into()
            }

            fn data_len(&self) -> usize {
                self.data().len()
            }
        }
        impl DType for u64 {
            fn dtype() -> safetensors::Dtype {
                safetensors::Dtype::U64
            }
        }
        impl DType for f32 {
            fn dtype() -> safetensors::Dtype {
                safetensors::Dtype::F32
            }
        }
        [
            ("s".to_owned(), DynView::new(Slice::new(s))),
            ("t".to_owned(), DynView::new(Slice::new(t))),
            ("c".to_owned(), DynView::new(Slice::new(c.as_slice()))),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use equator::assert;
    use faer::dyn_stack::GlobalPodBuffer;
    use rand::prelude::*;

    #[test]
    fn test_sct() {
        let rng = &mut StdRng::seed_from_u64(0);

        let m = 15;
        let n = 17;

        let A: Mat<f32> = faer::stats::StandardNormalMat { nrows: m, ncols: n }.sample(rng);
        let B: Mat<f32> = faer::stats::StandardNormalMat { nrows: m, ncols: n }.sample(rng);

        let mut cut_A = GreedyCuts::new(A.as_ref());
        let mut cut_B = GreedyCuts::new(B.as_ref());

        let mut mem = GlobalPodBuffer::new(cut_A.extend_scratch().unwrap());
        let stack = PodStack::new(&mut mem);

        for _ in 0..3 {
            cut_A.extend(37, rng, stack);
            cut_B.extend(15, rng, stack);
        }

        let err_A = (&cut_A.remainder_cis - (&A - cut_A.sct.expand())).norm_l2();
        let err_B = (&cut_B.remainder_cis - (&B - cut_B.sct.expand())).norm_l2();
        assert!(err_A < 1e-5);
        assert!(err_B < 1e-5);

        assert!(
            (&cut_A.remainder_cis + &cut_B.remainder_cis
                - (&A + &B - concat![cut_A.sct, cut_B.sct].expand()))
            .norm_l2()
                < 1e-5
        );
    }
}
