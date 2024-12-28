mod backend;

use clap::Parser;
use equator::assert;
use faer::{
    dyn_stack::{GlobalPodBuffer, PodStack},
    prelude::SolverCore,
    Col, Mat,
};
use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};
use signtensors::sct::{GreedyCuts, Sct, SctMut};

use backend::*;

#[derive(Debug, Parser)]
#[command(name = "Hadamard")]
#[command(about = "Approximates the identity matrix with signtensors", long_about = None)]
struct Args {
    /// The dimension
    #[arg(short = 'n')]
    nrows: usize,
    /// Print invariance error
    #[arg(short = 'e')]
    err: bool,
}

fn main() -> eyre::Result<()> {
    let Args { nrows, err } = Args::try_parse()?;
    let ncols = nrows;
    let eye: Mat<f32> = Mat::identity(nrows, ncols);
    let mut cuts = GreedyCuts::with_capacity(eye.as_ref(), nrows);
    let mut mem = GlobalPodBuffer::new(cuts.extend_scratch()?);
    let stack = PodStack::new(&mut mem);
    let rng = &mut StdRng::seed_from_u64(0);
    for i in 0..nrows {
        // dbg!(i, cuts.remainder_cis.squared_norm_l2() / nrows as f32);
        cuts.extend(1, rng, stack);
        // dbg!();
        // correct_scalars(&mut cuts);
        if err {
            print_invariance(&cuts)
        }
        correct_signs(&mut cuts, stack);
        if err {
            print_invariance(&cuts)
        }
        // todo!();
        // correct_remainder(&mut cuts);
        // if err {
        //     print_invariance(&cuts)
        // }
    }

    dbg!(cuts.remainder_cis.squared_norm_l2() / nrows as f32);
    print_invariance(&cuts);
    // let expanded_sct = cuts.sct.expand();
    // let error = eye.as_ref() - cuts.remainder_cis.as_ref() - expanded_sct;
    // dbg!(
    //     error.norm_l2(),
    //     cuts.sct
    //         .c
    //         .iter()
    //         .minmax_by(|u, v| u.partial_cmp(v).unwrap())
    // );
    // correct_scalars(&mut cuts);
    // dbg!(cuts.remainder_cis.squared_norm_l2() / nrows as f32);
    // let expanded_sct = cuts.sct.expand();
    // let error = eye.as_ref() - cuts.remainder_cis.as_ref() - expanded_sct;
    // dbg!(
    //     error.norm_l2(),
    //     cuts.sct
    //         .c
    //         .iter()
    //         .minmax_by(|u, v| u.partial_cmp(v).unwrap())
    // );
    // correct_scalars(&mut cuts);
    // dbg!(cuts.remainder_cis.squared_norm_l2() / nrows as f32);
    Ok(())
}

fn print_invariance(cuts: &GreedyCuts) {
    let nrows = cuts.nrows();
    let width = cuts.width();
    let remainder_true = faer::Mat::<f32>::identity(nrows, nrows) - cuts.sct.expand();
    let err = cuts.remainder_cis.as_ref() - remainder_true;
    println!("invariance (w = {width}): {} == 0", err.norm_l2())
}

enum RegenerateResult {
    Improved,
    EqualOrLess,
}

fn regenerate_triple(cuts: &mut GreedyCuts, i: usize, rng: &mut impl Rng) -> RegenerateResult {
    let GreedyCuts {
        sct,
        remainder_cis,
        remainder_trans,
    } = cuts;
    // let mut sct = sct.as_mut();
    // println!("{:?}", remainder_cis.as_ref());
    // dbg!(remainder_cis.as_ref(), sct.width());
    let scti = ith_cut(sct.as_mut(), i);
    let old_c = scti.c[0];
    *remainder_cis += scti.expand();
    remainder_trans.copy_from(remainder_cis.transpose());
    let mut one_cut = GreedyCuts::with_capacity(remainder_cis.as_ref(), 1);
    let mut mem = GlobalPodBuffer::new(one_cut.extend_scratch().unwrap());
    let stack = PodStack::new(&mut mem);
    one_cut.extend(1, rng, stack);
    // dbg!(
    //     i,
    //     &sct.c,
    //     sct.c[i] * sct.nrows() as f32,
    //     one_cut.sct.c[0] * sct.nrows() as f32
    // );
    let SctMut { s, c, t } = scti;
    let si = s.storage_mut().col_mut(0).try_as_slice_mut().unwrap();
    let ci = &mut c[..1];
    assert!(ci.len() == 1);
    let ti = t.storage_mut().col_mut(0).try_as_slice_mut().unwrap();
    let Sct {
        s,
        c,
        t,
        nrows,
        ncols,
    } = one_cut.sct;
    si.copy_from_slice(s.as_slice());
    ti.copy_from_slice(t.as_slice());
    ci.copy_from_slice(c.as_slice());
    remainder_cis.copy_from(one_cut.remainder_cis);
    remainder_trans.copy_from(one_cut.remainder_trans);
    if ci[0] < old_c {
        RegenerateResult::Improved
    } else {
        RegenerateResult::EqualOrLess
    }
}

fn ith_cut(sct: SctMut, i: usize) -> SctMut {
    let SctMut { s, c, t } = sct;
    let ci = &mut c[i..][..1];
    let (_, si) = s.split_at_col_mut(i);
    let (si, _) = si.split_at_col_mut(1);
    let (_, ti) = t.split_at_col_mut(i);
    let (ti, _) = ti.split_at_col_mut(1);
    assert!(all(ci.len() == 1, si.ncols() == 1, ti.ncols() == 1,));
    SctMut {
        s: si,
        c: ci,
        t: ti,
    }
}

// fn correct_scalars(cuts: &mut GreedyCuts) {
//     let GreedyCuts {
//         sct,
//         remainder_cis,
//         remainder_trans,
//     } = cuts;
//     let width = sct.width();
//     let Sct {
//         s,
//         c,
//         t,
//         nrows,
//         ncols,
//     } = sct;
//     let bit_dot = |s: &[u64], t: &[u64]| -> i32 {
//         let mut bit_diff = 0u32;
//         for (&si, &ti) in s.iter().zip(t) {
//             let sti = si ^ ti;
//             bit_diff += sti.count_ones();
//         }
//         *nrows as i32 - 2 * bit_diff as i32
//     };
//     let xtx: Mat<f32> = Mat::from_fn(width, width, |i, j| {
//         let stride = nrows.div_ceil(64);
//         let si = &s[stride * i..][..stride];
//         let sj = &s[stride * j..][..stride];
//         let sij = bit_dot(si, sj);
//         let ti = &t[stride * i..][..stride];
//         let tj = &t[stride * j..][..stride];
//         let tij = bit_dot(ti, tj);
//         (sij * tij) as f32
//     });
//     // dbg!(xtx.norm_l2());
//     // dbg!(xtx.as_ref());
//     let xti: Col<f32> = Col::from_fn(width, |i| {
//         let stride = nrows.div_ceil(64);
//         let si = &s[stride * i..][..stride];
//         let ti = &t[stride * i..][..stride];
//         let sti = bit_dot(si, ti);
//         sti as f32
//     });
//     // xtx.ll
//     let xtxi = xtx.cholesky(faer::Side::Upper).unwrap().inverse();
//     let c_new = xtxi * xti;
//     c.copy_from_slice(c_new.as_slice());
//     let new_remainder = Mat::<f32>::identity(*nrows, *ncols) - sct.expand();
//     remainder_cis.copy_from(new_remainder);
//     remainder_trans.copy_from(remainder_cis.transpose());
//     // todo!()
// }
