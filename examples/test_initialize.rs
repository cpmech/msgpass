use msgpass::{mpi_finalize, mpi_init, mpi_initialized, mpi_world_rank, mpi_world_size};
use std::env;

fn main() {
    let np = match env::var("CI") {
        Ok(_) => 1,
        Err(_) => 4,
    };

    println!("np = {}", np);

    assert!(!mpi_initialized().unwrap());
    mpi_init().unwrap();
    assert!(mpi_initialized().unwrap());
    assert!(mpi_world_rank().unwrap() <= np);
    assert_eq!(mpi_world_size().unwrap(), np);
    mpi_finalize().unwrap();
}
