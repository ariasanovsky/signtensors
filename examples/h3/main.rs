mod backend;

use clap::Parser;

use backend::*;
use faer::{
    dyn_stack::{GlobalPodBuffer, PodStack},
    Mat,
};
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug, Parser)]
#[command(name = "Hadamard")]
#[command(about = "Approximates the identity matrix with symmetric signtensors", long_about = None)]
struct Args {
    /// The dimension
    #[arg(short = 'n')]
    nrows: usize,
}

fn main() -> eyre::Result<()> {
    let Args { nrows } = Args::try_parse()?;
    let ncols = nrows;
    let eye: Mat<f32> = Mat::identity(nrows, ncols);
    // let mut cuts = SymmCuts::new(eye.as_ref());
    let mut cuts = SymmCuts::with_capacity(eye.as_ref(), nrows);
    let mut mem = GlobalPodBuffer::new(cuts.extend_scratch()?);
    let stack = PodStack::new(&mut mem);
    let rng = &mut StdRng::seed_from_u64(0);
    for i in 0..nrows {
        dbg!(i, cuts.remainder_symm.squared_norm_l2() / nrows as f32);
        cuts.extend(rng, stack);
        let true_remainder = eye.as_ref() - cuts.expand_symm_sct();
        let err = true_remainder - cuts.remainder_symm.as_ref();
        println!("(w = {i}): invariance = {}", err.norm_l2());
    }
    Ok(())
}
