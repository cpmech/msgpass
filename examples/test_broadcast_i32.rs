use msgpass::{mpi_finalize, mpi_init, mpi_world_rank, mpi_world_size, Communicator, StrError};

fn main() -> Result<(), StrError> {
    mpi_init()?;

    let rank = mpi_world_rank()?;
    let size = mpi_world_size()?;
    let mut comm = Communicator::new()?;

    let mut x = vec![0_i32; size];
    if rank == 0 {
        for i in 0..size {
            x[i] = 1000 + (i as i32);
        }
    }

    comm.broadcast_i32(0, &mut x)?;

    mpi_finalize()?;

    let mut correct = vec![0_i32; size];
    for i in 0..size {
        correct[i] = 1000 + (i as i32);
    }
    assert_eq!(&x, &correct);

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
