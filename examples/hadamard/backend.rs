use equator::assert;
use faer::dyn_stack::PodStack;
use reborrow::{Reborrow, ReborrowMut};
use signtensors::{
    inplace_sct::CutHelperMut,
    sct::{GreedyCuts, SctMut},
    SignMatMut,
};

const ALIGN: usize = 128;

pub(crate) fn correct_signs(cuts: &mut GreedyCuts, stack: &mut PodStack) -> usize {
    let (nrows, ncols) = cuts.shape();
    let width = cuts.width();

    let (s_signs_old, stack) = stack.make_aligned_raw::<u64>(nrows.div_ceil(64), ALIGN);
    let (t_signs_old, stack) = stack.make_aligned_raw::<u64>(ncols.div_ceil(64), ALIGN);
    let (s_signs, stack) = stack.make_aligned_raw::<u64>(nrows.div_ceil(64), ALIGN);
    let (t_signs, stack) = stack.make_aligned_raw::<u64>(ncols.div_ceil(64), ALIGN);
    let (mut t_image_half, stack) = faer::linalg::temp_mat_uninit::<f32>(nrows, 1, stack);
    let (mut s_image_half, stack) = faer::linalg::temp_mat_uninit::<f32>(ncols, 1, stack);

    let mut improvments = 0;

    let GreedyCuts {
        sct: sct_full,
        remainder_cis,
        remainder_trans,
    } = cuts;
    let mut sct_full = sct_full.as_mut();
    // dbg!(sct_full.width());

    for i in 0..width {
        let mut si = sct_full.s.rb_mut().split_at_col_mut(i);
        assert!(all(si.0.ncols() == i, si.1.ncols() == width - i,));
        let mut si = si.1.rb_mut().split_at_col_mut(1).0;
        assert!(si.ncols() == 1);
        let mut ti = sct_full.t.rb_mut().split_at_col_mut(i);
        assert!(all(ti.0.ncols() == i, ti.1.ncols() == width - i,));
        let mut ti = ti.1.rb_mut().split_at_col_mut(1).0;
        assert!(ti.ncols() == 1);
        let ci = &mut sct_full.c[i..][..1];
        // dbg!(ci[0]);
        let mut ci = faer::col::from_slice_mut(ci);

        assert!(all(si.ncols() == 1, ti.ncols() == 1, ci.nrows() == 1));

        let mut t_image_half = t_image_half.rb_mut().col_mut(0);
        let mut s_image_half = s_image_half.rb_mut().col_mut(0);

        s_signs.copy_from_slice(si.rb_mut().storage().col(0).try_as_slice().unwrap());
        s_signs_old.copy_from_slice(si.rb_mut().storage().col(0).try_as_slice().unwrap());
        t_signs.copy_from_slice(ti.rb_mut().storage().col(0).try_as_slice().unwrap());
        t_signs_old.copy_from_slice(ti.rb_mut().storage().col(0).try_as_slice().unwrap());

        t_image_half.fill(0.0);
        s_image_half.fill(0.0);

        let scti = SctMut {
            s: si.rb_mut(),
            c: ci.rb_mut().try_as_slice_mut().unwrap(),
            t: ti.rb_mut(),
        };

        // println!("remainder_cis = {remainder_cis:?}");

        *remainder_cis += scti.expand();
        // println!("remainder_cis = {remainder_cis:?}");

        remainder_trans.copy_from(remainder_cis.transpose());

        ci[0] = 0.0;
        // todo!("recalculate remainders with c_i = 0");
        signtensors::bitmagic::matvec_bit(
            nrows,
            ncols,
            t_image_half.rb_mut().try_as_slice_mut().unwrap(),
            remainder_cis.as_ref(),
            &bytemuck::cast_slice(t_signs)[..ncols.div_ceil(16)],
        );
        signtensors::bitmagic::matvec_bit(
            ncols,
            nrows,
            s_image_half.rb_mut().try_as_slice_mut().unwrap(),
            remainder_trans.as_ref(),
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
        let new_c = cut.cut_mat_inplace(
            remainder_cis.as_ref(),
            remainder_trans.as_ref(),
            si.rb_mut(),
            ci.rb_mut(),
            ti.rb_mut(),
            usize::MAX,
            stack,
        );
        // dbg!(new_c / (nrows * ncols) as f32);

        signtensors::bitmagic::matmul::mat_tmat_f32(
            remainder_cis.as_mut(),
            si.rb(),
            ti.rb(),
            ci.rb().try_as_slice().unwrap(),
        );
        remainder_trans
            .as_mut()
            .copy_from(remainder_cis.transpose());
        ci[0] = -ci[0];
    }

    correct_remainder(cuts);
    // todo!();
    improvments
}

pub(crate) fn correct_remainder(cuts: &mut GreedyCuts) {
    let GreedyCuts {
        sct,
        remainder_cis,
        remainder_trans,
    } = cuts;
    // todo!()
}

pub(crate) fn correct_scalars(cuts: &mut GreedyCuts) {
    let GreedyCuts {
        sct,
        remainder_cis,
        remainder_trans,
    } = cuts;
    todo!()
}
