use msgpass::*;

fn main() -> Result<(), StrError> {
    mpi_init()?;

    let rank = mpi_world_rank()?;
    let mut comm = Communicator::new()?;

    if rank == 0 {
        comm.abort(0)?;
    }

    mpi_finalize()?;

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
