use msgpass::*;

fn main() -> Result<(), StrError> {
    mpi_init()?;

    let rank = mpi_world_rank()?;
    let size = mpi_world_size()?;
    let mut comm = Communicator::new()?;

    const ROOT: i32 = 0;
    const TAG: i32 = 70;

    if rank == ROOT as usize {
        let x = vec![1.0, 2.0, 3.0];
        for to in 1..size {
            comm.send_f64(&x, to, TAG)?;
        }
        println!("{}: x = {:?}", rank, x);
    } else {
        let mut y = vec![0.0, 0.0, 0.0];
        comm.receive_f64(&mut y, ROOT, TAG)?;
        println!("{}: y = {:?}", rank, y);
    }

    mpi_finalize()
}
