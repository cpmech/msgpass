use msgpass::*;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), StrError> {
    mpi_init()?;

    let rank = mpi_world_rank()?;
    let mut comm = Communicator::new()?;

    if rank == 0 {
        thread::sleep(Duration::from_millis(20));
        println!("{} says hi << zero is always the last\n", rank);
    } else {
        println!("{} says hi", rank);
    }

    if rank == 0 {
        thread::sleep(Duration::from_millis(20));
    }

    comm.barrier()?;

    if rank == 0 {
        println!("{} says hi << now zero may not be the last", rank);
    } else {
        println!("{} says hi", rank);
    }

    mpi_finalize()?;

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
