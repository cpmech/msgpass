use msgpass::{mpi_finalize, mpi_init, mpi_initialized, mpi_world_rank, mpi_world_size, StrError};
use std::env;

fn main() -> Result<(), StrError> {
    let np = match env::var("CI") {
        Ok(_) => 1,
        Err(_) => 4,
    };

    assert!(!mpi_initialized()?);

    mpi_init()?;

    let rank = mpi_world_rank()?;
    let size = mpi_world_size()?;

    assert!(mpi_initialized()?);
    assert!(rank <= np);
    assert_eq!(size, np);

    mpi_finalize()?;

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
