mod backend;

use core::f32;
use std::{fs::File, io::Read};

use clap::Parser;
use cuts::{inplace_sct::CutHelper, SignMatMut};
use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use equator::assert;
use faer::{linalg::temp_mat_req, solvers::SolverCore, Col, ColMut, ColRef, Mat, MatMut, MatRef};
use image::{open, ImageBuffer, Rgb};
use itertools::Itertools;
use rand::{rngs::StdRng, SeedableRng};
use reborrow::{Reborrow, ReborrowMut};

use backend::*;

#[derive(Debug, Parser)]
#[command(name = "Angus")]
#[command(about = "Approximates an image with signed cuts", long_about = None)]
struct Args {
    /// Input directory containing `safetensors`
    #[arg(short = 'i')]
    input: std::path::PathBuf,
    /// Output directory for new `safetensors`
    #[arg(short = 'o')]
    output: std::path::PathBuf,
    /// The width
    #[arg(short = 'w')]
    width: usize,
    /// Render an image at step `{1, 1+s, 1+2s, ...}`
    #[arg(short = 's')]
    step: usize,
    /// How many iterative loops after increasing width
    #[arg(short = 'l')]
    loops: usize,
    /// What mode to run the program in
    #[arg(value_enum)]
    mode: Mode,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
enum Mode {
    /// Spans color axis with `c in RR` and `k: {-1, 1}^3`
    Signs,
    /// Spans color axis with `c in RR^3`
    Scalars,
} //05373980154234621

fn main() -> eyre::Result<()> {
    let Args {
        input,
        output,
        width,
        step,
        loops,
        mode,
    } = Args::try_parse()?;
    let stem = input
        .file_stem()
        .unwrap()
        .to_os_string()
        .into_string()
        .unwrap();
    let img = open(input)?.into_rgb8();
    let (nrows, ncols) = img.dimensions();
    let (nrows, ncols): (usize, usize) = (nrows as _, ncols as _);
    // let width_bits = 32 + 3 + nrows + ncols;
    // let nbytes = (width * width_bits).div_ceil(8);
    // dbg!(width, nbytes);
    let bytes = (0..3)
        .flat_map(|c| img.pixels().map(move |p| p.0[c]))
        .collect::<Vec<_>>();
    let bytes = RgbTensor::new(bytes, nrows, ncols);
    // let init_norm = bytes
    //     .data
    //     .iter()
    //     .map(|&b| {
    //         let b = b as i64;
    //         b * b
    //     })
    //     .sum::<i64>();
    let init_norm = 3 * nrows * ncols * 255 * 255;
    let a = bytes.clone().convert(|c| c as f32);

    let rng = &mut StdRng::seed_from_u64(0);
    let mut smat: SignMatrix = SignMatrix::new(nrows);
    let mut tmat: SignMatrix = SignMatrix::new(ncols);
    // let blowup = false;

    let mut rgb: RgbVector = match mode {
        Mode::Signs => RgbVector::blowup_with_capacity(width),
        Mode::Scalars => RgbVector::columns_with_capacity(width),
    };

    let mut r = a.clone();
    let mut mem = GlobalPodBuffer::new(
        StackReq::new::<u64>(Ord::max(nrows, ncols)).and(temp_mat_req::<f32>(1, 1).unwrap()),
    );
    let mut stack = PodStack::new(&mut mem);

    for w in 0..width {
        let r_gammas: [Mat<f32>; 4] = GAMMA.map(|k| r.combine_colors(&k));
        let cuts: [(Col<f32>, Col<f32>); 4] =
            core::array::from_fn(|i| greedy_cut(r_gammas[i].as_ref(), rng, stack.rb_mut()));
        let mut mats: [(SignMatrix, SignMatrix, RgbVector); 4] = core::array::from_fn(|i| {
            let mut smat = smat.clone();
            smat.push_col(cuts[i].0.as_slice());
            let mut tmat = tmat.clone();
            tmat.push_col(cuts[i].1.as_slice());
            match &rgb {
                RgbVector::Blowup { width, kmat, c } => {
                    let mut kmat = kmat.clone();
                    kmat.push_col(&GAMMA[i]);
                    let rgb = RgbVector::Blowup {
                        width: w + 1,
                        kmat,
                        c: Col::zeros(w + 1),
                    };
                    (smat, tmat, rgb)
                }
                RgbVector::Columns { width: _, r, g, b } => {
                    let rgb = RgbVector::Columns {
                        width: w + 1,
                        r: Col::zeros(w + 1),
                        g: Col::zeros(w + 1),
                        b: Col::zeros(w + 1),
                    };
                    (smat, tmat, rgb)
                }
            }
        });
        let regressions: [f32; 4] = core::array::from_fn(|i| {
            let (smat, tmat, rgb) = &mut mats[i];
            let anorm = regress(&a, smat, tmat, rgb.as_mut());
            anorm
        });
        let i_max = regressions
            .iter()
            .position_max_by(|a, b| a.partial_cmp(&b).unwrap())
            .unwrap();
        smat.push_col(cuts[i_max].0.as_slice());
        tmat.push_col(cuts[i_max].1.as_slice());
        match &mut rgb {
            RgbVector::Blowup { width, kmat, c } => {
                // let (new_smat, new_tmat, new_rgb) = &mats[i_max];
                kmat.push_col(GAMMA[i_max].as_slice());
                match mats[i_max].2.as_mut() {
                    RgbVectorMut::Blowup {
                        width: _,
                        kmat: _,
                        c: c_new,
                    } => {
                        *c = c_new.to_owned();
                        *width += 1;
                    }
                    _ => unreachable!(),
                };
            }
            RgbVector::Columns { width, r, g, b } => match mats[i_max].2.as_mut() {
                RgbVectorMut::Columns {
                    width: _,
                    r: r_new,
                    g: g_new,
                    b: b_new,
                } => {
                    *width += 1;
                    r.copy_from(r_new);
                    g.copy_from(g_new);
                    b.copy_from(b_new);
                }
                _ => unreachable!(),
            },
        }
        r = a.minus(&smat, &tmat, rgb.as_ref());
        // let improvements = (0, 0);
        let improvements = improve_signs_then_coefficients_repeatedly(
            &a,
            &mut r,
            &mut smat,
            &mut tmat,
            rgb.as_mut(),
            stack.rb_mut(),
            loops,
        );

        // let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
        // e.write_all(bytemuck::cast_slice(s_signs)).unwrap();
        // let foo = e.finish().unwrap();
        // let original_len = s_signs.len();
        // let new_len = foo.len();
        // let compression_rate = (s_signs.len() * 8) as f64 / foo.len() as f64;
        // println!("s compression rate = {original_len} * 8 / {new_len} = {compression_rate}");
        // dbg!(compression_rate);

        let approx = a.col() - r.col();
        let rel_error =
            ((total_error(approx.as_slice(), &bytes.data) as f64) / (init_norm as f64)).sqrt();
        let w = w + 1;
        let nbits = w * match mode {
            Mode::Signs => nrows + ncols + 3 + 32,
            Mode::Scalars => nrows + ncols + 3 * 32,
        };
        if true || w % step == 1 || w == width {
            let outpath = output.join(format!("{stem}-{w:04}.jpg"));
            let approx = a.col() - r.col();
            let output_image = ImageBuffer::from_fn(nrows as _, ncols as _, |i, j| {
                let i = i as usize;
                let j = j as usize;
                let ij = i + nrows * j;
                let rgb: [u8; 3] = core::array::from_fn(|c| to_u8(approx[c * nrows * ncols + ij]));
                Rgb(rgb)
            });
            output_image.save(outpath)?;
            let smol_outpath = output.join(format!("{stem}-{w:04}-signtensors-smol"));
            let mut smol_file = File::create(smol_outpath)?;
            write_compressed_tensors(&mut smol_file, &smat, &tmat, &rgb)?;
            let smol_file_bits = smol_file.metadata().unwrap().len() * 8;
            println!(
                "({}, {}, {}, {}, {}, {}, {}),",
                w,
                nbits,
                smol_file_bits,
                nbits as f64 / smol_file_bits as f64,
                improvements.0,
                improvements.1,
                rel_error
            );
        }
    }
    Ok(())
}

fn write_compressed_tensors(
    f: &mut File,
    smat: &SignMatrix,
    tmat: &SignMatrix,
    rgb: &RgbVector,
) -> eyre::Result<()> {
    use bytemuck::bytes_of;
    use flate2::{write::ZlibEncoder, Compression};
    use std::io::prelude::*;

    let smat = smat.as_mat_ref();
    let nrow_bytes = smat.nrows().div_ceil(8);
    let width = smat.ncols();
    let mut sbits = vec![0u8; nrow_bytes * width];
    for (col, col_bits) in smat.col_iter().zip(sbits.chunks_mut(nrow_bytes)) {
        // col.iter().enumerate().for_each(|(i, si)| {
        //     let pos = i / 8;
        //     let rem = i % 8;
        //     let bits = &mut col_bits[pos];
        //     if si.is_sign_negative() {
        //         *bits |= 1 << rem
        //     }
        // });
        for i in 0..col.nrows() {
            let pos = i / 8;
            let rem = i % 8;
            let bits = &mut col_bits[pos];
            let si = col[i];
            if i == 0 {
                if si.is_sign_negative() {
                    *bits |= 1 << rem
                }
            } else {
                let sprev = col[i - 1];
                if si.is_sign_negative() != sprev.is_sign_negative() {
                    *bits |= 1 << rem
                }
            }
        }
    }
    // dbg!(sbits.len());
    let mut z = ZlibEncoder::new(Vec::new(), Compression::best());
    z.write_all(&sbits)?;
    let s_smol = z.finish()?;
    // dbg!(s_smol.len());

    let tmat = tmat.as_mat_ref();
    let nrow_bytes = tmat.nrows().div_ceil(8);
    assert!(tmat.ncols() == width);
    let mut tbits = vec![0u8; nrow_bytes * width];
    for (col, col_bits) in tmat.col_iter().zip(tbits.chunks_mut(nrow_bytes)) {
        // col.iter().enumerate().for_each(|(i, ti)| {
        //     let pos = i / 8;
        //     let rem = i % 8;
        //     let bits = &mut col_bits[pos];
        //     if ti.is_sign_negative() {
        //         *bits |= 1 << rem
        //     }
        // });
        for i in 0..col.nrows() {
            let pos = i / 8;
            let rem = i % 8;
            let bits = &mut col_bits[pos];
            let ti = col[i];
            if i == 0 {
                if ti.is_sign_negative() {
                    *bits |= 1 << rem
                }
            } else {
                let tprev = col[i - 1];
                if ti.is_sign_negative() != tprev.is_sign_negative() {
                    *bits |= 1 << rem
                }
            }
        }
    }
    // dbg!(tbits.len());
    let mut z = ZlibEncoder::new(Vec::new(), Compression::best());
    z.write_all(&tbits)?;
    let t_smol = z.finish()?;
    // dbg!(t_smol.len());
    let (n_signtensors, nscalars, scalars_smol): (usize, usize, Vec<u8>) = match rgb.as_ref() {
        RgbVectorRef::Blowup { width: w, kmat, c } => {
            // assert!(all(

            // ));
            todo!("compress k & scalars")
        }
        RgbVectorRef::Columns { width: w, r, g, b } => {
            assert!(all(
                w == width,
                r.nrows() == width,
                g.nrows() == width,
                b.nrows() == width,
            ));
            let mut z = ZlibEncoder::new(Vec::new(), Compression::best());
            z.write_all(bytemuck::cast_slice(r.try_as_slice().unwrap()))?;
            z.write_all(bytemuck::cast_slice(g.try_as_slice().unwrap()))?;
            z.write_all(bytemuck::cast_slice(b.try_as_slice().unwrap()))?;
            (2, 3 * width, z.finish()?)
        }
    };
    f.write_all(bytes_of(&n_signtensors))?;
    f.write_all(bytes_of(&width))?;
    f.write_all(bytes_of(&smat.nrows()))?;
    f.write_all(bytes_of(&s_smol.len()))?;
    f.write_all(bytes_of(&tmat.nrows()))?;
    f.write_all(bytes_of(&t_smol.len()))?;
    match n_signtensors {
        2 => {
            f.write_all(bytes_of(&nscalars))?;
            f.write_all(bytes_of(&scalars_smol.len()))?;
            f.write_all(&s_smol)?;
            f.write_all(&t_smol)?;
            f.write_all(&scalars_smol)?;
        }
        3 => {
            todo!("deal with k");
            f.write_all(bytes_of(&nscalars))?;
            f.write_all(bytes_of(&scalars_smol.len()))?;
            f.write_all(&s_smol)?;
            f.write_all(&t_smol)?;
            todo!("deal with k");
            f.write_all(&scalars_smol)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}
