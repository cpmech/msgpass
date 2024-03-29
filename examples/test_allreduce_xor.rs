use msgpass::*;

fn main() -> Result<(), StrError> {
    mpi_init()?;

    let mut comm = Communicator::new()?;
    let rank = comm.rank()?;

    const N: usize = 3;

    let mut x_i32 = vec![0_i32; N];
    let mut x_i64 = vec![0_i64; N];
    let mut x_u32 = vec![0_u32; N];
    let mut x_u64 = vec![0_u64; N];
    let mut x_usz = vec![0_usize; N];
    let mut x_byt = vec![0_u8; N];

    if rank == 0 {
        for i in 0..N {
            x_i32[i] = i as i32;
            x_i64[i] = i as i64;
            x_u32[i] = i as u32;
            x_u64[i] = i as u64;
            x_usz[i] = i;
            x_byt[i] = i as u8;
        }
    }

    let mut y_i32 = vec![0_i32; N];
    let mut y_i64 = vec![0_i64; N];
    let mut y_u32 = vec![0_u32; N];
    let mut y_u64 = vec![0_u64; N];
    let mut y_usz = vec![0_usize; N];
    let mut y_byt = vec![0_u8; N];

    comm.allreduce_i32(&mut y_i32, &x_i32, MpiOpInt::Xor)?;
    comm.allreduce_i64(&mut y_i64, &x_i64, MpiOpInt::Xor)?;
    comm.allreduce_u32(&mut y_u32, &x_u32, MpiOpInt::Xor)?;
    comm.allreduce_u64(&mut y_u64, &x_u64, MpiOpInt::Xor)?;
    comm.allreduce_usize(&mut y_usz, &x_usz, MpiOpInt::Xor)?;
    comm.allreduce_bytes(&mut y_byt, &x_byt, MpiOpByte::Xor)?;

    let mut correct_i32 = vec![0_i32; N];
    let mut correct_i64 = vec![0_i64; N];
    let mut correct_u32 = vec![0_u32; N];
    let mut correct_u64 = vec![0_u64; N];
    let mut correct_usz = vec![0_usize; N];
    let mut correct_byt = vec![0_u8; N];
    for i in 1..N {
        correct_i32[i] = 1; // note that '2' becomes '1'
        correct_i64[i] = 1;
        correct_u32[i] = 1;
        correct_u64[i] = 1;
        correct_usz[i] = 1;
        correct_byt[i] = i as u8;
    }
    assert_eq!(&y_i32, &correct_i32);
    assert_eq!(&y_i64, &correct_i64);
    assert_eq!(&y_u32, &correct_u32);
    assert_eq!(&y_u64, &correct_u64);
    assert_eq!(&y_usz, &correct_usz);
    assert_eq!(&y_byt, &correct_byt);

    mpi_finalize()?;

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
