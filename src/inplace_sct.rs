use crate::{bitmagic, SignMatMut, SignMatRef};
use core::iter::zip;
use equator::assert;
use faer::{dyn_stack::PodStack, Col, ColMut, ColRef, MatRef};
use rand::prelude::*;
use reborrow::*;

pub struct CutHelper {
    pub t_signs_old: Box<[u64]>,
    pub s_signs_old: Box<[u64]>,
    pub t_signs: Box<[u64]>,
    pub s_signs: Box<[u64]>,
    pub t_image_half: Col<f32>,
    pub s_image_half: Col<f32>,
}

pub struct CutHelperMut<'a> {
    pub t_signs_old: &'a mut [u64],
    pub s_signs_old: &'a mut [u64],
    pub t_signs: &'a mut [u64],
    pub s_signs: &'a mut [u64],
    pub t_image_half: ColMut<'a, f32>,
    pub s_image_half: ColMut<'a, f32>,
}

impl CutHelperMut<'_> {
    // remainder (+ S * C * T^top)
    pub fn cut_mat(
        &mut self,
        remainder: MatRef<'_, f32>,
        remainder_transposed: MatRef<'_, f32>,
        S: SignMatMut<'_>,
        C: ColMut<'_, f32>,
        T: SignMatMut<'_>,
        rng: &mut dyn RngCore,
        max_iterations: usize,
        stack: &mut PodStack,
    ) -> f32 {
        let Self {
            t_signs,
            s_signs: _,
            t_image_half,
            s_image_half: _,
            t_signs_old,
            s_signs_old: _,
        } = self;
        assert!(all(
            remainder.row_stride() == 1,
            remainder_transposed.row_stride() == 1,
            C.nrows() > 0,
        ));

        t_signs_old.copy_from_slice(t_signs);
        rng.fill_bytes(bytemuck::cast_slice_mut(t_signs));

        mul_add_with_rank_update(
            t_image_half.rb_mut().try_as_slice_mut().unwrap(),
            remainder.rb(),
            S.rb(),
            C.rb(),
            T.rb(),
            bytemuck::cast_slice(t_signs),
            bytemuck::cast_slice(t_signs_old),
            stack,
        );

        self.cut_mat_inplace(
            remainder,
            remainder_transposed,
            S,
            C,
            T,
            max_iterations,
            stack,
        )
    }

    pub fn cut_mat_inplace(
        &mut self,
        remainder: MatRef<'_, f32>,
        remainder_transposed: MatRef<'_, f32>,
        S: SignMatMut<'_>,
        C: ColMut<'_, f32>,
        T: SignMatMut<'_>,
        max_iterations: usize,
        stack: &mut PodStack,
    ) -> f32 {
        let Self {
            t_signs,
            s_signs,
            t_image_half,
            s_image_half,
            t_signs_old,
            s_signs_old,
        } = self;
        assert!(all(
            remainder.row_stride() == 1,
            remainder_transposed.row_stride() == 1,
            C.nrows() > 0,
        ));

        let mut S = S;
        let mut T = T;
        let mut C = C;

        let mut cut = 0.0;

        let width = C.nrows() - 1;
        let (S, S_next) = S.rb_mut().split_at_col_mut(width);
        let (T, T_next) = T.rb_mut().split_at_col_mut(width);
        let (C, C_next) = C.rb_mut().split_at_mut(width);
        let S = S.rb();
        let T = T.rb();
        let C = C.rb();
        let C_next = C_next.get_mut(0);

        {
            let s_signs = bytemuck::cast_slice::<u64, u8>(s_signs);
            for (i, &t) in t_image_half
                .rb_mut()
                .try_as_slice()
                .unwrap()
                .iter()
                .enumerate()
            {
                let div = i / 8;
                let rem = i % 8;
                let sign = (((s_signs[div] >> rem) & 1 == 1) as u32) << 31;
                cut += f32::from_bits(t.to_bits() ^ sign);
            }
        }
        for _ in 0..max_iterations {
            let improved_s = improve_with_rank_update(
                remainder_transposed.rb(),
                T,
                C,
                S,
                t_image_half.as_ref(),
                bytemuck::cast_slice_mut(s_signs_old),
                bytemuck::cast_slice_mut(s_signs),
                s_image_half.as_mut(),
                &mut cut,
                stack,
            );

            if !improved_s {
                break;
            }
            let improved_t = improve_with_rank_update(
                remainder.rb(),
                S,
                C,
                T,
                s_image_half.as_ref(),
                bytemuck::cast_slice_mut(t_signs_old),
                bytemuck::cast_slice_mut(t_signs),
                t_image_half.as_mut(),
                &mut cut,
                stack,
            );

            if !improved_t {
                break;
            }
        }

        let normalization = remainder.nrows() * remainder.ncols();
        let normalized_cut = cut / normalization as f32;
        // remainder <- remainder - S * c * T^top
        // s_image <- s_image - T * c * S^top * S
        // t_image <- t_image - S * c * T^top * T

        {
            let t_signs = bytemuck::cast_slice::<u64, u8>(t_signs);
            let s_signs = bytemuck::cast_slice::<u64, u8>(s_signs);

            let k = cut / remainder.ncols() as f32;
            for (i, s) in s_image_half
                .rb_mut()
                .try_as_slice_mut()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                let div = i / 8;
                let rem = i % 8;
                let sign = (((t_signs[div] >> rem) & 1 == 1) as u32) << 31;
                *s -= f32::from_bits(k.to_bits() ^ sign)
            }

            let k = cut / remainder.nrows() as f32;
            for (i, t) in t_image_half
                .rb_mut()
                .try_as_slice_mut()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                let div = i / 8;
                let rem = i % 8;
                let sign = (((s_signs[div] >> rem) & 1 == 1) as u32) << 31;
                *t -= f32::from_bits(k.to_bits() ^ sign)
            }
        }
        *C_next = -normalized_cut;
        S_next
            .storage_mut()
            .col_mut(0)
            .try_as_slice_mut()
            .unwrap()
            .copy_from_slice(s_signs);
        T_next
            .storage_mut()
            .col_mut(0)
            .try_as_slice_mut()
            .unwrap()
            .copy_from_slice(t_signs);

        cut
    }
}

impl CutHelper {
    pub fn new(mat: MatRef<'_, f32>, mat_transposed: MatRef<'_, f32>) -> Self {
        let nrows = mat.nrows();
        let ncols = mat.ncols();

        let s_ones = vec![0u64; nrows.div_ceil(64)].into_boxed_slice();
        let t_ones = vec![0u64; ncols.div_ceil(64)].into_boxed_slice();

        CutHelper::new_with_st(mat, mat_transposed, s_ones, t_ones)
    }

    pub fn new_with_st(
        mat: MatRef<'_, f32>,
        mat_transposed: MatRef<'_, f32>,
        s_signs: Box<[u64]>,
        t_signs: Box<[u64]>,
    ) -> Self {
        let (nrows, ncols) = mat.shape();
        let mut s_image_half = Col::<f32>::zeros(ncols);
        let mut t_image_half = Col::<f32>::zeros(nrows);

        bitmagic::matvec_bit(
            ncols,
            nrows,
            s_image_half.as_slice_mut(),
            mat_transposed,
            &bytemuck::cast_slice(&s_signs)[..nrows.div_ceil(16)],
        );
        bitmagic::matvec_bit(
            nrows,
            ncols,
            t_image_half.as_slice_mut(),
            mat,
            &bytemuck::cast_slice(&t_signs)[..ncols.div_ceil(16)],
        );
        s_image_half *= faer::scale(0.5f32);
        t_image_half *= faer::scale(0.5f32);

        Self {
            t_signs: t_signs.clone(),
            s_signs: s_signs.clone(),
            t_image_half,
            s_image_half,
            t_signs_old: t_signs,
            s_signs_old: s_signs,
        }
    }

    pub fn as_mut(&mut self) -> CutHelperMut<'_> {
        CutHelperMut {
            t_signs_old: &mut self.t_signs_old,
            s_signs_old: &mut self.s_signs_old,
            t_signs: &mut self.t_signs,
            s_signs: &mut self.s_signs,
            t_image_half: self.t_image_half.as_mut(),
            s_image_half: self.s_image_half.as_mut(),
        }
    }

    // remainder (+ S * C * T^top)
    pub fn cut_mat(
        &mut self,
        remainder: MatRef<'_, f32>,
        remainder_transposed: MatRef<'_, f32>,
        S: SignMatMut<'_>,
        C: ColMut<'_, f32>,
        T: SignMatMut<'_>,
        rng: &mut dyn RngCore,
        max_iterations: usize,
        stack: &mut PodStack,
    ) -> f32 {
        self.as_mut().cut_mat(
            remainder,
            remainder_transposed,
            S,
            C,
            T,
            rng,
            max_iterations,
            stack,
        )
    }

    pub fn cut_mat_inplace(
        &mut self,
        remainder: MatRef<'_, f32>,
        remainder_transposed: MatRef<'_, f32>,
        S: SignMatMut<'_>,
        C: ColMut<'_, f32>,
        T: SignMatMut<'_>,
        max_iterations: usize,
        stack: &mut PodStack,
    ) -> f32 {
        self.as_mut().cut_mat_inplace(
            remainder,
            remainder_transposed,
            S,
            C,
            T,
            max_iterations,
            stack,
        )
    }
}

pub(crate) fn sparse_matvec(acc: &mut [f32], two_mat: MatRef<'_, f32>, diff_indices: &[u64]) {
    struct Impl<'a> {
        acc: &'a mut [f32],
        two_mat: MatRef<'a, f32>,
        diff_indices: &'a [u64],
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                acc,
                two_mat,
                diff_indices,
            } = self;
            for &j in diff_indices {
                let negate = (j >> 63) == 1;
                let j = (j & ((1 << 63) - 1)) as usize;
                let col = two_mat.col(j).try_as_slice().unwrap();

                if negate {
                    for (acc, &src) in zip(&mut *acc, col) {
                        *acc -= src;
                    }
                } else {
                    for (acc, &src) in zip(&mut *acc, col) {
                        *acc += src;
                    }
                }
            }
        }
    }
    pulp::Arch::new().dispatch(Impl {
        acc,
        two_mat,
        diff_indices,
    })
}

pub(crate) fn sparse_matvec_sign(
    acc: &mut [f32],
    two_mat: MatRef<'_, f32>,
    diff_indices: &[usize],
    negate: bool,
) {
    struct Impl<'a> {
        acc: &'a mut [f32],
        two_mat: MatRef<'a, f32>,
        diff_indices: &'a [usize],
        negate: bool,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                acc,
                two_mat,
                diff_indices,
                negate,
            } = self;
            for &j in diff_indices {
                let j = j & ((1 << 63) - 1);
                let col = two_mat.col(j).try_as_slice().unwrap();

                if negate {
                    for (acc, &src) in zip(&mut *acc, col) {
                        *acc -= src;
                    }
                } else {
                    for (acc, &src) in zip(&mut *acc, col) {
                        *acc += src;
                    }
                }
            }
        }
    }
    pulp::Arch::new().dispatch(Impl {
        acc,
        two_mat,
        diff_indices,
        negate,
    })
}

// acc += two * (mat + S * C * T^top) * (x_new - x_old)
// 1. acc += two * mat * (x_new - x_old)
// 2. acc += two * S * C * T^top * (x_new - x_old)
fn mul_add_with_rank_update(
    acc: &mut [f32],
    two_mat: MatRef<'_, f32>,
    S: SignMatRef<'_>,
    C: ColRef<'_, f32>,
    T: SignMatRef<'_>,
    x_new: &[u8],
    x_old: &[u8],
    stack: &mut PodStack,
) {
    let width = T.ncols();
    let n = two_mat.ncols();
    let (diff_indices, stack) = stack.make_raw::<u64>(n);
    let diff_indices = mul_add(acc, two_mat, x_new, x_old, diff_indices);

    // y = T^top * (x_new - x_old)
    let (y, _) = faer::linalg::temp_mat_uninit::<f32>(width, 1, stack);
    let y = y.col_mut(0).try_as_slice_mut().unwrap();
    for (y, t) in core::iter::zip(&mut *y, T.storage_as::<u8>().col_iter()) {
        let col = t.try_as_slice().unwrap();
        let mut acc = 0i64;
        for &i in diff_indices {
            let negate = (i >> 63) == 1;
            let i = i & ((1 << 63) - 1);
            let i = i as usize;
            let negate2 = (col[i / 8] >> (i % 8)) & 1 == 1;
            if negate == negate2 {
                acc += 1;
            } else {
                acc -= 1;
            }
        }
        *y = acc as f32;
    }
    // y = 2.0 * C * y
    for (y, &c) in zip(&mut *y, C.try_as_slice().unwrap()) {
        *y *= 2.0 * c;
    }
    // acc += S * y
    bitmagic::matvec::matvec_f32(acc, S, y);
}

// acc += two * (mat + S * C * T^top) * (x_new - x_old)
// 1. acc += two * mat * (x_new - x_old)
// 2. acc += two * S * C * T^top * (x_new - x_old)
fn mul_add<'out>(
    acc: &mut [f32],
    two_mat: MatRef<'_, f32>,
    x_new: &[u8],
    x_old: &[u8],
    diff_indices: &'out mut [u64],
) -> &'out [u64] {
    let n = two_mat.ncols();

    let mut pos = 0usize;
    for j in 0..n {
        let s_neg = (x_new[j / 8] >> (j % 8)) & 1 == 1;
        let s_neg_old = (x_old[j / 8] >> (j % 8)) & 1 == 1;
        if s_neg != s_neg_old {
            diff_indices[pos] = ((s_neg as u64) << 63) | j as u64;
            pos += 1;
        }
    }

    let diff_indices = &diff_indices[..pos];

    sparse_matvec(acc, two_mat, diff_indices);
    diff_indices
}

pub(crate) fn improve_with_rank_update(
    two_mat: MatRef<'_, f32>,
    S: SignMatRef<'_>,
    C: ColRef<'_, f32>,
    T: SignMatRef<'_>,
    s_image: ColRef<'_, f32>,
    t_signs_old: &mut [u8],
    t_signs: &mut [u8],
    mut t_image: ColMut<'_, f32>,
    cut: &mut f32,
    stack: &mut PodStack,
) -> bool {
    let new_cut = s_image.norm_l1();
    if new_cut <= *cut {
        return false;
    } else {
        *cut = new_cut
    }

    t_signs_old.copy_from_slice(t_signs.rb());
    let s_image = s_image.try_as_slice().unwrap();
    for (i, t) in t_signs.iter_mut().enumerate() {
        let mut sign = 0u8;
        for idx in 0..8 {
            if 8 * i + idx < s_image.len() {
                sign |= ((s_image[8 * i + idx].to_bits() >> 31) as u8) << idx;
            }
        }
        *t = sign;
    }

    let t_image = t_image.rb_mut().try_as_slice_mut().unwrap();

    mul_add_with_rank_update(t_image, two_mat, S, C, T, t_signs, t_signs_old, stack);

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{
        dyn_stack::{GlobalPodBuffer, StackReq},
        linalg::temp_mat_req,
        Mat,
    };

    #[test]
    fn test_cuts_regular() {
        let dim = 128;
        let nrows = dim;
        let ncols = dim;
        let blocksize = 32;

        let rng = &mut StdRng::seed_from_u64(0);

        let A: Mat<f32> = faer::stats::StandardNormalMat { nrows, ncols }.sample(rng);
        let init_norm = A.squared_norm_l2();

        let mut remainder: Mat<f32> = A.to_owned();
        let mut remainder_transposed = A.transpose().to_owned();
        let mut helper = CutHelper::new(remainder.as_ref(), remainder_transposed.as_ref());
        let mut S = vec![0u64; (nrows.div_ceil(64)) * blocksize].into_boxed_slice();
        let mut T = vec![0u64; (ncols.div_ceil(64)) * blocksize].into_boxed_slice();
        let mut C = Col::<f32>::zeros(blocksize);

        let bf16_residual = Mat::from_fn(nrows, ncols, |i, j| {
            A[(i, j)] - half::bf16::from_f32(A[(i, j)]).to_f32()
        })
        .squared_norm_l2();

        let mut S = SignMatMut::from_storage(
            faer::mat::from_column_major_slice_mut(&mut S, nrows.div_ceil(64), blocksize),
            nrows,
        );
        let mut T = SignMatMut::from_storage(
            faer::mat::from_column_major_slice_mut(&mut T, ncols.div_ceil(64), blocksize),
            ncols,
        );
        let mut how_full = 0usize;
        let mut mem = GlobalPodBuffer::new(
            StackReq::new::<u64>(Ord::max(nrows, ncols))
                .and(temp_mat_req::<f32>(blocksize, 1).unwrap()),
        );
        let stack = PodStack::new(&mut mem);

        let mut full_S: Vec<u64> = vec![];
        let mut full_T: Vec<u64> = vec![];
        let mut full_C: Vec<f32> = vec![];

        let mut remainder_norm = init_norm;

        let mut iter = 0;
        while iter < 20000 {
            if how_full == blocksize {
                {
                    let remainder = remainder.as_mut();
                    crate::bitmagic::matmul::mat_tmat_f32(remainder, S.rb(), T.rb(), C.as_slice());
                }

                remainder_transposed
                    .as_mut()
                    .copy_from(remainder.transpose());
                remainder_norm = remainder.squared_norm_l2() / 4.0;

                for k in 0..blocksize {
                    full_S.extend_from_slice(S.rb().storage().col(k).try_as_slice().unwrap());
                    full_T.extend_from_slice(T.rb().storage().col(k).try_as_slice().unwrap());
                    full_C.push(-C[k]);
                }
                how_full = 0;
            }

            how_full += 1;
            let cut = helper.cut_mat(
                remainder.as_ref(),
                remainder_transposed.as_ref(),
                S.rb_mut().split_at_col_mut(how_full).0,
                C.get_mut(..how_full),
                T.rb_mut().split_at_col_mut(how_full).0,
                rng,
                usize::MAX,
                stack,
            );

            remainder_norm -= (cut * cut) / (nrows * ncols) as f32;
            if remainder_norm <= bf16_residual {
                {
                    let remainder = remainder.as_mut();
                    crate::bitmagic::matmul::mat_tmat_f32(
                        remainder,
                        S.rb_mut().split_at_col_mut(how_full).0.rb(),
                        T.rb_mut().split_at_col_mut(how_full).0.rb(),
                        &C.as_slice()[..how_full],
                    );
                }
                remainder_transposed
                    .as_mut()
                    .copy_from(remainder.transpose());

                for k in 0..how_full {
                    full_S.extend_from_slice(S.rb().storage().col(k).try_as_slice().unwrap());
                    full_T.extend_from_slice(T.rb().storage().col(k).try_as_slice().unwrap());
                    full_C.push(-C[k]);
                }

                break;
            }

            iter += 1;
        }
        dbg!(iter);
        dbg!(remainder_norm / init_norm);

        let break_even_point = iter;

        let sct_bytes = ((nrows / 8 + ncols / 8 + 4) * break_even_point) as f64;
        let f32_bytes = (nrows * ncols * 4) as f64;
        let f16_bytes = (nrows * ncols * 2) as f64;

        eprintln!("dim: ({nrows}*{ncols}), break even at: {break_even_point}, f32 compression rate: {:.3}, bf16 compression rate: {:.3}", sct_bytes / f32_bytes, sct_bytes / f16_bytes);
    }
}
