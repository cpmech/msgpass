use msgpass::*;

fn main() -> Result<(), StrError> {
    mpi_init()?;

    let rank = mpi_world_rank()?;
    let size = mpi_world_size()?;
    let mut comm = Communicator::new()?;

    const N: usize = 3;

    let mut x_i32 = vec![0_i32; N];
    let mut x_i64 = vec![0_i64; N];
    let mut x_u32 = vec![0_u32; N];
    let mut x_u64 = vec![0_u64; N];
    let mut x_usz = vec![0_usize; N];
    let mut x_f32 = vec![0_f32; N];
    let mut x_f64 = vec![0_f64; N];

    for i in 0..N {
        x_i32[i] = 1000 + (i as i32);
        x_i64[i] = 1000 + (i as i64);
        x_u32[i] = 1000 + (i as u32);
        x_u64[i] = 1000 + (i as u64);
        x_usz[i] = 1000 + i;
        x_f32[i] = 1000.0 + (i as f32);
        x_f64[i] = 1000.0 + (i as f64);
    }

    let mut y_i32 = vec![0_i32; N];
    let mut y_i64 = vec![0_i64; N];
    let mut y_u32 = vec![0_u32; N];
    let mut y_u64 = vec![0_u64; N];
    let mut y_usz = vec![0_usize; N];
    let mut y_f32 = vec![0_f32; N];
    let mut y_f64 = vec![0_f64; N];

    comm.reduce_i32(0, &mut y_i32, &x_i32, MpiOp::Sum)?;
    comm.reduce_i64(0, &mut y_i64, &x_i64, MpiOp::Sum)?;
    comm.reduce_u32(0, &mut y_u32, &x_u32, MpiOp::Sum)?;
    comm.reduce_u64(0, &mut y_u64, &x_u64, MpiOp::Sum)?;
    comm.reduce_usize(0, &mut y_usz, &x_usz, MpiOp::Sum)?;
    comm.reduce_f32(0, &mut y_f32, &x_f32, MpiOp::Sum)?;
    comm.reduce_f64(0, &mut y_f64, &x_f64, MpiOp::Sum)?;

    if rank == 0 {
        let mut correct_i32 = vec![0_i32; N];
        let mut correct_i64 = vec![0_i64; N];
        let mut correct_u32 = vec![0_u32; N];
        let mut correct_u64 = vec![0_u64; N];
        let mut correct_usz = vec![0_usize; N];
        let mut correct_f32 = vec![0_f32; N];
        let mut correct_f64 = vec![0_f64; N];
        for i in 0..N {
            correct_i32[i] = (size as i32) * x_i32[i];
            correct_i64[i] = (size as i64) * x_i64[i];
            correct_u32[i] = (size as u32) * x_u32[i];
            correct_u64[i] = (size as u64) * x_u64[i];
            correct_usz[i] = size * x_usz[i];
            correct_f32[i] = (size as f32) * x_f32[i];
            correct_f64[i] = (size as f64) * x_f64[i];
        }
        // println!("size = {}", size);
        // println!("{}: x = {:?}", rank, x_i32);
        // println!("{}: y = {:?} = {} * x", rank, y_i32, size);
        assert_eq!(&y_i32, &correct_i32);
        assert_eq!(&y_i64, &correct_i64);
        assert_eq!(&y_u32, &correct_u32);
        assert_eq!(&y_u64, &correct_u64);
        assert_eq!(&y_usz, &correct_usz);
        assert_eq!(&y_f32, &correct_f32);
        assert_eq!(&y_f64, &correct_f64);
    } else {
        for i in 0..N {
            assert_eq!(y_i32[i], 0);
            assert_eq!(y_i64[i], 0);
            assert_eq!(y_u32[i], 0);
            assert_eq!(y_u64[i], 0);
            assert_eq!(y_usz[i], 0);
            assert_eq!(y_f32[i], 0.0);
            assert_eq!(y_f64[i], 0.0);
        }
    }

    mpi_finalize()?;

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
