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
    let mut x_byt = vec![0_u8; N];

    for i in 0..N {
        x_i32[i] = 1000 + (rank as i32);
        x_i64[i] = 1000 + (rank as i64);
        x_u32[i] = 1000 + (rank as u32);
        x_u64[i] = 1000 + (rank as u64);
        x_usz[i] = 1000 + rank;
        x_f32[i] = 1000.0 + (rank as f32);
        x_f64[i] = 1000.0 + (rank as f64);
        x_c32[i] = Complex32::new(1000.0 + (rank as f32), 1000.0 + (rank as f32));
        x_c64[i] = Complex64::new(1000.0 + (rank as f64), 1000.0 + (rank as f64));
        x_byt[i] = rank as u8;
    }

    let mut y_i32 = vec![0_i32; N * size];
    let mut y_i64 = vec![0_i64; N * size];
    let mut y_u32 = vec![0_u32; N * size];
    let mut y_u64 = vec![0_u64; N * size];
    let mut y_usz = vec![0_usize; N * size];
    let mut y_f32 = vec![0_f32; N * size];
    let mut y_f64 = vec![0_f64; N * size];
    let mut y_c32 = vec![Complex32::new(0.0, 0.0); N * size];
    let mut y_c64 = vec![Complex64::new(0.0, 0.0); N * size];
    let mut y_byt = vec![0_u8; N * size];

    comm.allgather_i32(&mut y_i32, &x_i32)?;
    comm.allgather_i64(&mut y_i64, &x_i64)?;
    comm.allgather_u32(&mut y_u32, &x_u32)?;
    comm.allgather_u64(&mut y_u64, &x_u64)?;
    comm.allgather_usize(&mut y_usz, &x_usz)?;
    comm.allgather_f32(&mut y_f32, &x_f32)?;
    comm.allgather_f64(&mut y_f64, &x_f64)?;
    comm.allgather_c32(&mut y_c32, &x_c32)?;
    comm.allgather_c64(&mut y_c64, &x_c64)?;
    comm.allgather_bytes(&mut y_byt, &x_byt)?;

    let mut correct_i32 = vec![0_i32; N * size];
    let mut correct_i64 = vec![0_i64; N * size];
    let mut correct_u32 = vec![0_u32; N * size];
    let mut correct_u64 = vec![0_u64; N * size];
    let mut correct_usz = vec![0_usize; N * size];
    let mut correct_f32 = vec![0_f32; N * size];
    let mut correct_f64 = vec![0_f64; N * size];
    let mut correct_c32 = vec![Complex32::new(0.0, 0.0); N * size];
    let mut correct_c64 = vec![Complex64::new(0.0, 0.0); N * size];
    let mut correct_byt = vec![0_u8; N * size];
    for j in 0..size {
        for i in 0..N {
            let n = i + N * j;
            correct_i32[n] = 1000 + (j as i32);
            correct_i64[n] = 1000 + (j as i64);
            correct_u32[n] = 1000 + (j as u32);
            correct_u64[n] = 1000 + (j as u64);
            correct_usz[n] = 1000 + j;
            correct_f32[n] = 1000.0 + (j as f32);
            correct_f64[n] = 1000.0 + (j as f64);
            correct_c32[n] = Complex32::new(1000.0 + (j as f32), 1000.0 + (j as f32));
            correct_c64[n] = Complex64::new(1000.0 + (j as f64), 1000.0 + (j as f64));
            correct_byt[n] = j as u8;
        }
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
    assert_eq!(&y_byt, &correct_byt);

    mpi_finalize()?;

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
