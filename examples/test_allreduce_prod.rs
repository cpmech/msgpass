use msgpass::*;
use num_complex::{Complex32, Complex64};

fn factorial(num: usize) -> usize {
    (1..=num).product()
    // (2..=num).into_iter().fold(1, |acc, x| acc * x)
}

fn complex32_fact(num: usize) -> Complex32 {
    (2..=num).into_iter().fold(Complex32::new(1.0, 1.0), |acc, x| acc * Complex32::new(x as f32, x as f32))
}

fn complex64_fact(num: usize) -> Complex64 {
    (2..=num).into_iter().fold(Complex64::new(1.0, 1.0), |acc, x| acc * Complex64::new(x as f64, x as f64))
}

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
        x_i32[i] = rank as i32 + 1;
        x_i64[i] = rank as i64 + 1;
        x_u32[i] = rank as u32 + 1;
        x_u64[i] = rank as u64 + 1;
        x_usz[i] = rank + 1;
        x_f32[i] = rank as f32 + 1.0;
        x_f64[i] = rank as f64 + 1.0;
        x_c32[i] = Complex32::new(rank as f32 + 1.0, rank as f32 + 1.0);
        x_c64[i] = Complex64::new(rank as f64 + 1.0, rank as f64 + 1.0);
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

    comm.allreduce_i32(&mut y_i32, &x_i32, MpiOpInt::Prod)?;
    comm.allreduce_i64(&mut y_i64, &x_i64, MpiOpInt::Prod)?;
    comm.allreduce_u32(&mut y_u32, &x_u32, MpiOpInt::Prod)?;
    comm.allreduce_u64(&mut y_u64, &x_u64, MpiOpInt::Prod)?;
    comm.allreduce_usize(&mut y_usz, &x_usz, MpiOpInt::Prod)?;
    comm.allreduce_f32(&mut y_f32, &x_f32, MpiOpReal::Prod)?;
    comm.allreduce_f64(&mut y_f64, &x_f64, MpiOpReal::Prod)?;
    comm.allreduce_c32(&mut y_c32, &x_c32, MpiOpComplex::Prod)?;
    comm.allreduce_c64(&mut y_c64, &x_c64, MpiOpComplex::Prod)?;

    let f = factorial(size);
    let f_c32 = complex32_fact(size);
    let f_c64 = complex64_fact(size);
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
        correct_i32[i] = f as i32;
        correct_i64[i] = f as i64;
        correct_u32[i] = f as u32;
        correct_u64[i] = f as u64;
        correct_usz[i] = f;
        correct_f32[i] = f as f32;
        correct_f64[i] = f as f64;
        correct_c32[i] = f_c32;
        correct_c64[i] = f_c64;
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
