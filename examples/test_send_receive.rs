use msgpass::*;

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

    const TAG_I32: i32 = 10;
    const TAG_I64: i32 = 20;
    const TAG_U32: i32 = 30;
    const TAG_U64: i32 = 40;
    const TAG_USZ: i32 = 50;
    const TAG_F32: i32 = 60;
    const TAG_F64: i32 = 70;

    if rank == 0 {
        for to in 1..size {
            comm.send_i32(&x_i32, to, TAG_I32)?;
            comm.send_i64(&x_i64, to, TAG_I64)?;
            comm.send_u32(&x_u32, to, TAG_U32)?;
            comm.send_u64(&x_u64, to, TAG_U64)?;
            comm.send_usize(&x_usz, to, TAG_USZ)?;
            comm.send_f32(&x_f32, to, TAG_F32)?;
            comm.send_f64(&x_f64, to, TAG_F64)?;
        }
    } else {
        comm.receive_i32(&mut y_i32, 0, TAG_I32)?;
        assert_eq!(&y_i32, &x_i32);
        let res = comm.get_receive_status();
        assert_eq!(res, (0, TAG_I32, 0));

        comm.receive_i64(&mut y_i64, 0, TAG_I64)?;
        assert_eq!(&y_i64, &x_i64);
        let res = comm.get_receive_status();
        assert_eq!(res, (0, TAG_I64, 0));

        comm.receive_u32(&mut y_u32, 0, TAG_U32)?;
        assert_eq!(&y_u32, &x_u32);
        let res = comm.get_receive_status();
        assert_eq!(res, (0, TAG_U32, 0));

        comm.receive_u64(&mut y_u64, 0, TAG_U64)?;
        assert_eq!(&y_u64, &x_u64);
        let res = comm.get_receive_status();
        assert_eq!(res, (0, TAG_U64, 0));

        comm.receive_usize(&mut y_usz, 0, TAG_USZ)?;
        assert_eq!(&y_usz, &x_usz);
        let res = comm.get_receive_status();
        assert_eq!(res, (0, TAG_USZ, 0));

        comm.receive_f32(&mut y_f32, 0, TAG_F32)?;
        assert_eq!(&y_f32, &x_f32);
        let res = comm.get_receive_status();
        assert_eq!(res, (0, TAG_F32, 0));

        comm.receive_f64(&mut y_f64, 0, TAG_F64)?;
        assert_eq!(&y_f64, &x_f64);
        let res = comm.get_receive_status();
        assert_eq!(res, (0, TAG_F64, 0));
    }

    mpi_finalize()?;

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
