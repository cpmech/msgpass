# Thin wrapper to a Message Passing Interface (MPI)

MsgPass (Message Passing) is a thin Rust wrapper to OpenMPI. We consider a small subset of MPI functions. This subset will grow as our projects require more functionality. We implement (by hand) C functions that Rust can easily call using the FFI (in the `c_code` directory).

We try to test all functions as much as possible, but test coverage could be better. The tests must be called with `mpiexec`, thus it is easy to use the `run-tests.bash` script.

## Requirements

```bash
sudo apt install libopenmpi-dev
```

## Examples

The example below (available in the `examples` directory) will send an array from ROOT to all the other processors.

```rust
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
```

Running the code above with `mpiexec -np 4 ex_send_receive` (see `run-examples.bash`), we get an output similar to the one below:

```text
### ex_send_receive ######################################################
2: y = [1.0, 2.0, 3.0]
0: x = [1.0, 2.0, 3.0]
3: y = [1.0, 2.0, 3.0]
1: y = [1.0, 2.0, 3.0]
```
