use msgpass::{mpi_finalize, mpi_init, mpi_initialized, mpi_world_rank, mpi_world_size};

fn main() {
    assert!(!mpi_initialized().unwrap());
    mpi_init().unwrap();
    assert!(mpi_initialized().unwrap());
    assert!(mpi_world_rank().unwrap() <= 4);
    assert_eq!(mpi_world_size().unwrap(), 4);
    mpi_finalize().unwrap();
}
