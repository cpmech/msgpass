# Thin wrapper to a Message Passing Interface (MPI)

[![Test on Linux](https://github.com/cpmech/msgpass/actions/workflows/test_on_linux.yml/badge.svg)](https://github.com/cpmech/msgpass/actions/workflows/test_on_linux.yml)
[![Test on macOS](https://github.com/cpmech/msgpass/actions/workflows/test_on_macos.yml/badge.svg)](https://github.com/cpmech/msgpass/actions/workflows/test_on_macos.yml)

## Contents

* [Introduction](#introduction)
* [Crates](#crates)
* [Installation on Debian/Ubuntu/Linux](#installation)
* [Installation on macOS](#macos)
* [Examples](#examples)
* [Todo list](#todo)

## <a name="introduction"></a> Introduction

MsgPass (Message Passing) is a thin Rust wrapper to MPI. We consider a small subset of MPI functions. This subset will grow as our projects require more functionality. We implement (by hand) C functions that Rust can easily call using the FFI (in the `c_code` directory).

We try to test all functions as much as possible, but test coverage could be better. The tests must be called with `mpiexec`, thus it is easy to use the `run-tests.bash` script.

> [!NOTE]
> Unlike the MPI standard, we use `mpi_init` to initialize the simulation with multiple threads. Thus, our `mpi_init` function calls `MPI_Init_thread` with `MPI_THREAD_MULTIPLE`. On the other hand, our `mpi_init_single_thread` calls `MPI_Init`.

## <a name="crates"></a> Crates

TODO

## <a name="installation"></a> Installation on Debian/Ubuntu/Linux

On **Ubuntu/Linux**, install OpenMPI or MPICH (requires an environment variable). For instance,

```bash
sudo apt install libopenmpi-dev
```

or

```bash
sudo apt install libmpich-dev
```

For MPICH, the following environment variable is required:

```bash
export MSGPASS_USE_MPICH=1
```

## <a name="macos"></a> Installation on macOS

On **macOS**, install the following packages:


```bash
brew install llvm@13 open-mpi
```

Also, export the following environment variable:

```bash
export echo TMPDIR=/tmp
```

## <a name="examples"></a> Examples

See also:

* [examples](https://github.com/cpmech/msgpass/tree/main/examples)

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

## <a name="todo"></a> Todo list

- [x] Implement basic functionality
    - [x] Initialize and finalize
    - [x] Abort and barrier
- [ ] Wrap more MPI functions
    - [x] Implement reduce
    - [x] Implement allreduce
    - [x] Implement send/receive
    - [ ] Implement scatter/gather
