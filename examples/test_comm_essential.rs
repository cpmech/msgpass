use msgpass::*;
use std::env;

fn main() -> Result<(), StrError> {
    let np = match env::var("CI") {
        Ok(_) => 2,
        Err(_) => 4,
    };

    mpi_init()?;

    let mut comm = Communicator::new()?;
    let rank = comm.rank()?;
    let size = comm.size()?;

    assert!(rank <= np);
    assert_eq!(size, np);

    if np == 2 {
        let mut sub = Communicator::new_subset(&[0])?;
        if rank == 0 {
            assert_eq!(sub.rank()?, 0);
            assert_eq!(sub.size()?, 1);
        }
    } else {
        let mut sub = Communicator::new_subset(&[1, 3])?;
        if rank == 1 || rank == 3 {
            let sub_rank = sub.rank()?;
            let sub_size = sub.size()?;
            assert!(sub_rank == 0 || sub_rank == 1);
            assert_eq!(sub_size, 2);
        }
    }

    mpi_finalize()?;

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
