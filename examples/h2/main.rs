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
        sct: Sct::new(nrows, ncols, 1),
        remainder: eye.clone(),
    };
    let rng = &mut StdRng::seed_from_u64(0);
    let mut curr_err = cuts.remainder.squared_norm_l2() / nrows as f32;
    for i in 0..nrows {
        // println!(
        //     "R * [1,1,-1,-1] = {:?}",
        //     cuts.remainder.as_ref() * faer::col![1.0f32, 1.0, -1.0, -1.0]
        // );
        // println!(
        //     "R * [-1,1,-1,-1] = {:?}",
        //     cuts.remainder.as_ref() * faer::col![-1.0f32, 1.0, -1.0, -1.0]
        // );
        dbg!(i, cuts.remainder.squared_norm_l2() / nrows as f32);
        cuts.extend(rng);
        let new_err = cuts.remainder.squared_norm_l2() / nrows as f32;
        // println!("R_{i} = {:?} has err = {new_err}", cuts.remainder.as_ref());
        if new_err >= curr_err {
            panic!()
        }
        // todo!();
        // dbg!();
        // correct_scalars(&mut cuts);
        // if err {
        //     print_invariance(&cuts)
        // }
        // correct_signs(&mut cuts, stack);
        // if err {
        //     print_invariance(&cuts)
        // }
        // // todo!();
        // correct_remainder(&mut cuts);
        // if err {
        //     print_invariance(&cuts)
        // }
    }
    dbg!(cuts.remainder.squared_norm_l2() / nrows as f32);

    Ok(())
}
