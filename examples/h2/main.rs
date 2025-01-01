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
#[command(about = "Approximates the identity matrix with symmetric signtensors", long_about = None)]
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
    // let mut cuts = GreedyCuts::with_capacity(eye.as_ref(), nrows);
    let mut cuts = GreedyScs {
        sct: Sct::new(nrows, ncols, 0),
        remainder: eye.clone(),
    };
    let rng = &mut StdRng::seed_from_u64(0);
    let mut curr_err = cuts.remainder.squared_norm_l2() / nrows as f32;
    for i in 0..nrows {
        print_invariance(&cuts);
        dbg!(i, cuts.remainder.squared_norm_l2() / nrows as f32);
        cuts.extend(rng);
        let new_err = cuts.remainder.squared_norm_l2() / nrows as f32;
        // println!("R_{i} = {:?} has err = {new_err}", cuts.remainder.as_ref());
        if new_err >= curr_err {
            panic!()
        }
    }
    print_invariance(&cuts);

    dbg!(cuts.remainder.squared_norm_l2() / nrows as f32);
    loop {
        let mut best_error = cuts.remainder.squared_norm_l2() / nrows as f32;
        let mut improved = false;
        for i in 0..nrows {
            dbg!(i, cuts.remainder.squared_norm_l2() / nrows as f32);
            // print_invariance(&cuts);
            cuts.correct_signs(i);
            // let new_error = cuts.remainder.squared_norm_l2() / nrows as f32;
            // if new_error < best_error {
            //     improved = true;
            //     best_error = new_error;
            // }
            // print_invariance(&cuts);
        }
        correct_remainder(&mut cuts);
        correct_scalars(&mut cuts);
        let new_error = cuts.remainder.squared_norm_l2() / nrows as f32;
        if new_error < best_error {
            improved = true;
            best_error = new_error;
        }
        print_invariance(&cuts);
        if !improved {
            break;
        }
    }
    dbg!(cuts.remainder.squared_norm_l2() / nrows as f32);
    println!("s = {:?}", cuts.sct.s);
    Ok(())
}

fn print_invariance(cuts: &GreedyScs) {
    let nrows = cuts.shape().0;
    let width = cuts.width();
    let remainder_true = faer::Mat::<f32>::identity(nrows, nrows) - cuts.sct.expand();
    let err = cuts.remainder.as_ref() - remainder_true;
    println!("invariance (w = {width}): {} == 0", err.norm_l2())
}

pub(crate) fn correct_scalars(cuts: &mut GreedyScs) {
    let GreedyScs { sct, remainder } = cuts;
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
    println!("xtx = {:?}", xtx.as_ref());
    let svd = xtx.thin_svd();
    println!("sings = {:?}", svd.s_diagonal());
    let xtxi = svd.pseudoinverse();
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

pub(crate) fn correct_remainder(cuts: &mut GreedyScs) {
    // let GreedyCuts {
    //     sct,
    //     remainder_cis,
    //     remainder_trans,
    // } = cuts;
    let nrows = cuts.shape().0;
    let new_remainder = Mat::<f32>::identity(nrows, nrows) - cuts.sct.expand();
    cuts.remainder.copy_from(new_remainder);
}
