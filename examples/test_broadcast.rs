use msgpass::*;

fn main() -> Result<(), StrError> {
    mpi_init()?;

    let mut comm = Communicator::new()?;
    let rank = comm.rank()?;
    let size = comm.size()?;

    let mut x_i32 = vec![0_i32; size];
    let mut x_i64 = vec![0_i64; size];
    let mut x_u32 = vec![0_u32; size];
    let mut x_u64 = vec![0_u64; size];
    let mut x_usz = vec![0_usize; size];
    let mut x_f32 = vec![0_f32; size];
    let mut x_f64 = vec![0_f64; size];

    let mut correct_i32 = x_i32.clone();
    let mut correct_i64 = x_i64.clone();
    let mut correct_u32 = x_u32.clone();
    let mut correct_u64 = x_u64.clone();
    let mut correct_usz = x_usz.clone();
    let mut correct_f32 = x_f32.clone();
    let mut correct_f64 = x_f64.clone();

    // correct
    for i in 0..size {
        correct_i32[i] = 1000 + (i as i32);
        correct_i64[i] = 1000 + (i as i64);
        correct_u32[i] = 1000 + (i as u32);
        correct_u64[i] = 1000 + (i as u64);
        correct_usz[i] = 1000 + i;
        correct_f32[i] = 1000.0 + (i as f32);
        correct_f64[i] = 1000.0 + (i as f64);
    }

    if rank == 0 {
        x_i32 = correct_i32.clone();
        x_i64 = correct_i64.clone();
        x_u32 = correct_u32.clone();
        x_u64 = correct_u64.clone();
        x_usz = correct_usz.clone();
        x_f32 = correct_f32.clone();
        x_f64 = correct_f64.clone();
    }

    comm.broadcast_i32(0, &mut x_i32)?;
    comm.broadcast_i64(0, &mut x_i64)?;
    comm.broadcast_u32(0, &mut x_u32)?;
    comm.broadcast_u64(0, &mut x_u64)?;
    comm.broadcast_usize(0, &mut x_usz)?;
    comm.broadcast_f32(0, &mut x_f32)?;
    comm.broadcast_f64(0, &mut x_f64)?;

    mpi_finalize()?;

    assert_eq!(&x_i32, &correct_i32);
    assert_eq!(&x_i64, &correct_i64);
    assert_eq!(&x_u32, &correct_u32);
    assert_eq!(&x_u64, &correct_u64);
    assert_eq!(&x_usz, &correct_usz);
    assert_eq!(&x_f32, &correct_f32);
    assert_eq!(&x_f64, &correct_f64);

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
