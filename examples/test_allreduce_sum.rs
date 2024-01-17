use msgpass::*;
use num_complex::{Complex32, Complex64};

fn main() -> Result<(), StrError> {
    mpi_init()?;

    let mut comm = Communicator::new()?;
    let rank = comm.rank()?;
    let size = comm.size()?;

    const N: usize = 3;

    let mut x_i32 = vec![0_i32; N];
    let mut x_i64 = vec![0_i64; N];
    let mut x_u32 = vec![0_u32; N];
    let mut x_u64 = vec![0_u64; N];
    let mut x_usz = vec![0_usize; N];
    let mut x_f32 = vec![0_f32; N];
    let mut x_f64 = vec![0_f64; N];
    let mut x_c32 = vec![Complex32::new(0.0, 0.0); N];
    let mut x_c64 = vec![Complex64::new(0.0, 0.0); N];

    for i in 0..N {
        x_i32[i] = 1000 + (i as i32);
        x_i64[i] = 1000 + (i as i64);
        x_u32[i] = 1000 + (i as u32);
        x_u64[i] = 1000 + (i as u64);
        x_usz[i] = 1000 + i;
        x_f32[i] = 1000.0 + (i as f32);
        x_f64[i] = 1000.0 + (i as f64);
        x_c32[i] = Complex32::new(1000.0 + (i as f32), 1000.0 + (i as f32));
        x_c64[i] = Complex64::new(1000.0 + (i as f64), 1000.0 + (i as f64));
    }

    let mut y_i32 = vec![0_i32; N];
    let mut y_i64 = vec![0_i64; N];
    let mut y_u32 = vec![0_u32; N];
    let mut y_u64 = vec![0_u64; N];
    let mut y_usz = vec![0_usize; N];
    let mut y_f32 = vec![0_f32; N];
    let mut y_f64 = vec![0_f64; N];
    let mut y_c32 = vec![Complex32::new(0.0, 0.0); N];
    let mut y_c64 = vec![Complex64::new(0.0, 0.0); N];

    comm.allreduce_i32(&mut y_i32, &x_i32, MpiOpInt::Sum)?;
    comm.allreduce_i64(&mut y_i64, &x_i64, MpiOpInt::Sum)?;
    comm.allreduce_u32(&mut y_u32, &x_u32, MpiOpInt::Sum)?;
    comm.allreduce_u64(&mut y_u64, &x_u64, MpiOpInt::Sum)?;
    comm.allreduce_usize(&mut y_usz, &x_usz, MpiOpInt::Sum)?;
    comm.allreduce_f32(&mut y_f32, &x_f32, MpiOpReal::Sum)?;
    comm.allreduce_f64(&mut y_f64, &x_f64, MpiOpReal::Sum)?;
    comm.allreduce_c32(&mut y_c32, &x_c32, MpiOpComplex::Sum)?;
    comm.allreduce_c64(&mut y_c64, &x_c64, MpiOpComplex::Sum)?;

    let mut correct_i32 = vec![0_i32; N];
    let mut correct_i64 = vec![0_i64; N];
    let mut correct_u32 = vec![0_u32; N];
    let mut correct_u64 = vec![0_u64; N];
    let mut correct_usz = vec![0_usize; N];
    let mut correct_f32 = vec![0_f32; N];
    let mut correct_f64 = vec![0_f64; N];
    let mut correct_c32 = vec![Complex32::new(0.0, 0.0); N];
    let mut correct_c64 = vec![Complex64::new(0.0, 0.0); N];
    for i in 0..N {
        correct_i32[i] = (size as i32) * x_i32[i];
        correct_i64[i] = (size as i64) * x_i64[i];
        correct_u32[i] = (size as u32) * x_u32[i];
        correct_u64[i] = (size as u64) * x_u64[i];
        correct_usz[i] = size * x_usz[i];
        correct_f32[i] = (size as f32) * x_f32[i];
        correct_f64[i] = (size as f64) * x_f64[i];
        correct_c32[i] = Complex32::new((size as f32) * x_f32[i], (size as f32) * x_f32[i]);
        correct_c64[i] = Complex64::new((size as f64) * x_f64[i], (size as f64) * x_f64[i]);
    }
    assert_eq!(&y_i32, &correct_i32);
    assert_eq!(&y_i64, &correct_i64);
    assert_eq!(&y_u32, &correct_u32);
    assert_eq!(&y_u64, &correct_u64);
    assert_eq!(&y_usz, &correct_usz);
    assert_eq!(&y_f32, &correct_f32);
    assert_eq!(&y_f64, &correct_f64);
    assert_eq!(&y_c32, &correct_c32);
    assert_eq!(&y_c64, &correct_c64);

    mpi_finalize()?;

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
