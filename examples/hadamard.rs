use clap::Parser;
use equator::assert;
use faer::{
    dyn_stack::{GlobalPodBuffer, PodStack},
    Col, Mat,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use reborrow::ReborrowMut;
use signtensors::sct::{GreedyCuts, Sct, SctMut, SctRef};

#[derive(Debug, Parser)]
#[command(name = "Hadamard")]
#[command(about = "Approximates the identity matrix with signtensors", long_about = None)]
struct Args {
    /// The dimension
    #[arg(short = 'n')]
    nrows: usize,
}

fn main() -> eyre::Result<()> {
    let Args { nrows } = Args::try_parse()?;
    let ncols = nrows;
    let eye: Mat<f32> = Mat::identity(nrows, ncols);
    let mut cuts = GreedyCuts::with_capacity(eye.as_ref(), nrows);
    let mut mem = GlobalPodBuffer::new(cuts.extend_scratch()?);
    let stack = PodStack::new(&mut mem);
    let rng = &mut StdRng::seed_from_u64(0);
    for i in 0..nrows {
        cuts.extend(1, rng, stack);
        // loop {
        //     let mut improvements = 0;
        //     for j in 0..i {
        //         match regenerate_triple(&mut cuts, j, rng) {
        //             RegenerateResult::Improved => improvements += 1,
        //             RegenerateResult::EqualOrLess => {}
        //         }
        //     }
        //     if improvements == 0 {
        //         break;
        //     } else {
        //         println!("w = {i}, {improvements} new improvements")
        //     }
        // }
    }

    dbg!(cuts.remainder_cis.squared_norm_l2() / nrows as f32);
    let expanded_sct = cuts.sct.expand();
    let error = eye.as_ref() - cuts.remainder_cis.as_ref() - expanded_sct;
    dbg!(error.norm_l2());
    // correct_scalars(&mut cuts);
    // dbg!(cuts.remainder_cis.squared_norm_l2() / nrows as f32);
    Ok(())
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
    // println!("{:?}", scti.expand());
    // let rk = remainder_cis.as_ref() + scti.expand();
    // println!("R_{i} = {:?}", rk.as_ref());
    // let mut sct_dummy = sct.rb_mut().to_owned();
    // sct_dummy.c[i] = 0.0f32;
    // let other_rk: Mat<f32> =
    //     faer::Mat::<f32>::identity(remainder_cis.nrows(), remainder_cis.ncols())
    //         - sct_dummy.expand();
    // println!("(also) R_{i} = {:?}", other_rk.as_ref());
    // panic!();
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
    if scti.c[0] < one_cut.sct.c[0] {
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
//     dbg!(xtx.norm_l2());
//     dbg!(xtx.as_ref());
//     let xti: Col<f32> = Col::from_fn(width, |i| {
//         let stride = nrows.div_ceil(64);
//         let si = &s[stride * i..][..stride];
//         let ti = &t[stride * i..][..stride];
//         let sti = bit_dot(si, ti);
//         sti as f32
//     });
//     // xtx.ll
//     let xtxi = xtx.lblt(faer::Side::Upper);
//     dbg!(xtxi.)
//     todo!()
// }
