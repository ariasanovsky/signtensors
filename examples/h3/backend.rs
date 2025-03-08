use equator::assert;
use faer::{
    dyn_stack::{PodStack, SizeOverflow, StackReq},
    Mat, MatRef,
};
use rand::RngCore;
use reborrow::{Reborrow, ReborrowMut};
use signtensors::{
    inplace_sct::CutHelperMut,
    sct::{Sct, SctMut, SctRef},
};

pub(crate) struct SymmCuts {
    pub remainder_symm: Mat<f32>,
    pub sct_cis: Sct,
}

const ALIGN: usize = 128;

impl SymmCuts {
    const BLOCKSIZE: usize = 32;

    // pub(crate) fn new(a: MatRef<f32>) -> Self {
    //     todo!()
    // }

    pub fn with_capacity(mat: MatRef<'_, f32>, width_capacity: usize) -> Self {
        let (nrows, ncols) = mat.shape();
        Self {
            remainder_symm: mat.to_owned(),
            sct_cis: Sct {
                s: Vec::with_capacity(nrows.div_ceil(64) * width_capacity),
                c: Vec::with_capacity(width_capacity),
                t: Vec::with_capacity(ncols.div_ceil(64) * width_capacity),
                nrows,
                ncols,
            },
        }
    }

    pub fn extend_scratch(&self) -> Result<StackReq, SizeOverflow> {
        let (nrows, ncols) = self.remainder_symm.shape();
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

    pub(crate) fn extend(&mut self, rng: &mut dyn RngCore, stack: &mut PodStack) {
        let (nrows, ncols) = self.remainder_symm.shape();
        let cur = self.sct_cis.width();

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

        signtensors::bitmagic::matvec_bit(
            nrows,
            ncols,
            t_image_half.rb_mut().try_as_slice_mut().unwrap(),
            self.remainder_symm.as_ref(),
            &bytemuck::cast_slice(t_signs)[..ncols.div_ceil(16)],
        );
        signtensors::bitmagic::matvec_bit(
            ncols,
            nrows,
            s_image_half.rb_mut().try_as_slice_mut().unwrap(),
            self.remainder_symm.as_ref(),
            &bytemuck::cast_slice(s_signs)[..nrows.div_ceil(16)],
        );
        s_image_half *= 0.5_f32;
        t_image_half *= 0.5_f32;

        let mut cut = CutHelperMut {
            t_signs_old,
            s_signs_old,
            t_signs,
            s_signs,
            t_image_half,
            s_image_half,
        };

        self.sct_cis.s.resize(nrows.div_ceil(64) * (cur + 1), 0);
        self.sct_cis.t.resize(ncols.div_ceil(64) * (cur + 1), 0);
        self.sct_cis.c.resize(1 + cur, 0.0);

        let SctMut { mut s, c, mut t } = self.sct_cis.as_mut();

        let (_, mut s) = s.rb_mut().split_at_col_mut(cur);
        let (_, mut t) = t.rb_mut().split_at_col_mut(cur);
        let (_, c) = c.split_at_mut(cur);
        assert!(all(
            s.ncols() == 1, //
            t.ncols() == 1, //
            c.len() == 1,   //
        ));

        cut.cut_mat(
            self.remainder_symm.as_ref(),
            self.remainder_symm.as_ref(),
            s.rb_mut().split_at_col_mut(1).0,
            faer::col::from_slice_mut(&mut c[..1]),
            t.rb_mut().split_at_col_mut(1).0,
            rng,
            usize::MAX,
            stack,
        );

        let bit_dot = |s: &[u64], t: &[u64]| -> i32 {
            let mut bit_diff = 0u32;
            for (&si, &ti) in s.iter().zip(t) {
                let sti = si ^ ti;
                bit_diff += sti.count_ones();
            }
            nrows as i32 - 2 * bit_diff as i32
        };

        let s_dot_t = bit_dot(
            s.rb().storage().col(0).try_as_slice().unwrap(), //
            t.rb().storage().col(0).try_as_slice().unwrap(), //
        );
        c[0] *= (nrows * nrows) as f32 / (nrows * nrows + (s_dot_t * s_dot_t) as usize) as f32;

        signtensors::bitmagic::matmul::mat_tmat_f32(
            self.remainder_symm.as_mut(),
            s.rb().split_at_col(1).0,
            t.rb().split_at_col(1).0,
            &c[..1],
        );

        signtensors::bitmagic::matmul::mat_tmat_f32(
            self.remainder_symm.as_mut(),
            t.rb().split_at_col(1).0,
            s.rb().split_at_col(1).0,
            &c[..1],
        );

        c[0] = -c[0];

        // dbg!(
        //     self.sct_cis.s.as_slice(),
        //     self.sct_cis.t.as_slice(),
        //     self.sct_cis.c.as_slice(),
        //     self.remainder_symm.squared_norm_l2() / nrows as f32
        // );
        // println!("R = {:?}", self.remainder_symm.as_ref())
    }

    pub(crate) fn expand_symm_sct(&self) -> Mat<f32> {
        let Self {
            remainder_symm: _,
            sct_cis,
        } = self;
        let mat_cis = sct_cis.expand();
        let SctRef { s, c, t } = sct_cis.as_ref();
        let sct_trans = SctRef { s: t, c, t: s };
        mat_cis + sct_trans.expand()
    }
}
