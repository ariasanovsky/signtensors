use equator::assert;
use faer::{dyn_stack::PodStack, Col, Mat};
use reborrow::{Reborrow, ReborrowMut};
use signtensors::{
    inplace_sct::CutHelperMut,
    sct::{GreedyCuts, Sct, SctMut},
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

    // correct_remainder(cuts);
    // todo!();
    improvments
}

pub(crate) fn correct_remainder(cuts: &mut GreedyCuts) {
    // let GreedyCuts {
    //     sct,
    //     remainder_cis,
    //     remainder_trans,
    // } = cuts;
    let nrows = cuts.nrows();
    let new_remainder = Mat::<f32>::identity(nrows, nrows) - cuts.sct.expand();
    cuts.remainder_cis.copy_from(new_remainder);
    cuts.remainder_trans
        .copy_from(cuts.remainder_cis.transpose());
}

pub(crate) fn correct_scalars(cuts: &mut GreedyCuts) {
    let GreedyCuts {
        sct,
        remainder_cis,
        remainder_trans,
    } = cuts;
    let GreedyCuts {
        sct,
        remainder_cis,
        remainder_trans,
    } = cuts;
    let width = sct.width();
    let Sct {
        s,
        c,
        t,
        nrows,
        ncols,
    } = sct;
    let bit_dot = |s: &[u64], t: &[u64]| -> i32 {
        let mut bit_diff = 0u32;
        for (&si, &ti) in s.iter().zip(t) {
            let sti = si ^ ti;
            bit_diff += sti.count_ones();
        }
        *nrows as i32 - 2 * bit_diff as i32
    };
    let xtx: Mat<f32> = Mat::from_fn(width, width, |i, j| {
        let stride = nrows.div_ceil(64);
        let si = &s[stride * i..][..stride];
        let sj = &s[stride * j..][..stride];
        let sij = bit_dot(si, sj);
        let ti = &t[stride * i..][..stride];
        let tj = &t[stride * j..][..stride];
        let tij = bit_dot(ti, tj);
        (sij * tij) as f32
    });
    let xtxi = xtx.thin_svd().pseudoinverse();
    let xti: Col<f32> = Col::from_fn(width, |i| {
        let stride = nrows.div_ceil(64);
        let si = &s[stride * i..][..stride];
        let ti = &t[stride * i..][..stride];
        let sti = bit_dot(si, ti);
        sti as f32
    });
    let c_new = xtxi * xti;
    c.copy_from_slice(c_new.as_slice());
    correct_remainder(cuts);
}

pub(crate) fn zap_bottom(cuts: &mut GreedyCuts, k: usize, rng: &mut impl rand::Rng) {
    let mut values: Vec<(usize, f32)> = cuts
        .sct
        .c
        .iter()
        .enumerate()
        .map(|(i, &c)| (i, c))
        .collect();
    values.sort_by(|(i, c), (j, d)| c.partial_cmp(d).unwrap().then(i.cmp(j)));
    let sct = &mut cuts.sct;
    let stride = sct.nrows().div_ceil(64);
    for i in 0..k {
        let i = values[i].0;
        sct.c[i] = 0.0;
        // let si = &mut sct.s[stride * i..][..stride];
        // rng.fill_bytes(bytemuck::cast_slice_mut(si));
        // let ti = &mut sct.s[stride * i..][..stride];
        // rng.fill_bytes(bytemuck::cast_slice_mut(ti));
    }
    correct_remainder(cuts);
    // println!("min: {:?}, max: {:?}", values[0], values.last().unwrap());
    // todo!()
}
