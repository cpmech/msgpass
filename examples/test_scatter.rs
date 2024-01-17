use msgpass::*;

// See figure in https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/

fn main() -> Result<(), StrError> {
    mpi_init()?;

    let mut comm = Communicator::new()?;
    let rank = comm.rank()?;
    let size = comm.size()?;

    const N: usize = 3;

    let mut y_i32 = vec![0_i32; N];
    let mut y_i64 = vec![0_i64; N];
    let mut y_u32 = vec![0_u32; N];
    let mut y_u64 = vec![0_u64; N];
    let mut y_usz = vec![0_usize; N];
    let mut y_f32 = vec![0_f32; N];
    let mut y_f64 = vec![0_f64; N];

    if rank == 0 {
        let mut x_i32 = vec![0_i32; N * size];
        let mut x_i64 = vec![0_i64; N * size];
        let mut x_u32 = vec![0_u32; N * size];
        let mut x_u64 = vec![0_u64; N * size];
        let mut x_usz = vec![0_usize; N * size];
        let mut x_f32 = vec![0_f32; N * size];
        let mut x_f64 = vec![0_f64; N * size];
        for j in 0..size {
            for i in 0..N {
                let n = i + N * j;
                x_i32[n] = 1000 + (j as i32);
                x_i64[n] = 1000 + (j as i64);
                x_u32[n] = 1000 + (j as u32);
                x_u64[n] = 1000 + (j as u64);
                x_usz[n] = 1000 + j;
                x_f32[n] = 1000.0 + (j as f32);
                x_f64[n] = 1000.0 + (j as f64);
            }
        }
        comm.scatter_i32(0, &mut y_i32, Some(&x_i32))?;
        comm.scatter_i64(0, &mut y_i64, Some(&x_i64))?;
        comm.scatter_u32(0, &mut y_u32, Some(&x_u32))?;
        comm.scatter_u64(0, &mut y_u64, Some(&x_u64))?;
        comm.scatter_usize(0, &mut y_usz, Some(&x_usz))?;
        comm.scatter_f32(0, &mut y_f32, Some(&x_f32))?;
        comm.scatter_f64(0, &mut y_f64, Some(&x_f64))?;
    } else {
        comm.scatter_i32(0, &mut y_i32, None)?;
        comm.scatter_i64(0, &mut y_i64, None)?;
        comm.scatter_u32(0, &mut y_u32, None)?;
        comm.scatter_u64(0, &mut y_u64, None)?;
        comm.scatter_usize(0, &mut y_usz, None)?;
        comm.scatter_f32(0, &mut y_f32, None)?;
        comm.scatter_f64(0, &mut y_f64, None)?;
    }

    let mut correct_i32 = vec![0_i32; N];
    let mut correct_i64 = vec![0_i64; N];
    let mut correct_u32 = vec![0_u32; N];
    let mut correct_u64 = vec![0_u64; N];
    let mut correct_usz = vec![0_usize; N];
    let mut correct_f32 = vec![0_f32; N];
    let mut correct_f64 = vec![0_f64; N];
    for i in 0..N {
        correct_i32[i] = 1000 + (rank as i32);
        correct_i64[i] = 1000 + (rank as i64);
        correct_u32[i] = 1000 + (rank as u32);
        correct_u64[i] = 1000 + (rank as u64);
        correct_usz[i] = 1000 + rank;
        correct_f32[i] = 1000.0 + (rank as f32);
        correct_f64[i] = 1000.0 + (rank as f64);
    }
    assert_eq!(&y_i32, &correct_i32);
    assert_eq!(&y_i64, &correct_i64);
    assert_eq!(&y_u32, &correct_u32);
    assert_eq!(&y_u64, &correct_u64);
    assert_eq!(&y_usz, &correct_usz);
    assert_eq!(&y_f32, &correct_f32);
    assert_eq!(&y_f64, &correct_f64);

    mpi_finalize()?;

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
