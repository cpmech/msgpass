# Thin wrapper to a Message Passing Interface (MPI)

[![Documentation](https://docs.rs/msgpass/badge.svg)](https://docs.rs/msgpass)
[![Test on macOS](https://github.com/cpmech/msgpass/actions/workflows/test_on_macos.yml/badge.svg)](https://github.com/cpmech/msgpass/actions/workflows/test_on_macos.yml)
[![Test on Linux](https://github.com/cpmech/msgpass/actions/workflows/test_on_linux.yml/badge.svg)](https://github.com/cpmech/msgpass/actions/workflows/test_on_linux.yml)
[![Test on Linux with Intel MPI](https://github.com/cpmech/msgpass/actions/workflows/test_on_linux_intel_mpi.yml/badge.svg)](https://github.com/cpmech/msgpass/actions/workflows/test_on_linux_intel_mpi.yml)

## Contents

* [Introduction](#introduction)
* [Installation on Debian/Ubuntu/Linux](#installation)
* [Installation on macOS](#macos)
* [Setting Cargo.toml](#cargo)
* [Examples](#examples)
* [Todo list](#todo)

## <a name="introduction"></a> Introduction

MsgPass (Message Passing) is a thin Rust wrapper to MPI. We consider a small subset of MPI functions. This subset will grow as our projects require more functionality. We implement (by hand) C functions that Rust can easily call using the FFI (in the `c_code` directory).

We try to test all functions as much as possible, but test coverage could be better. The tests must be called with `mpiexec`, thus it is easy to use the `run-tests.bash` script.

**Documentation:**

- [![Documentation](https://docs.rs/msgpass/badge.svg)](https://docs.rs/msgpass)

## <a name="installation"></a> Installation on Debian/Ubuntu/Linux

On **Ubuntu/Linux**, install OpenMPI, MPICH, or Intel MPI. For instance,

```bash
sudo apt install libopenmpi-dev
```

or

```bash
sudo apt install libmpich-dev
```

or

```bash
bash ./zscripts/install-intel-mpi-debian.bash
```

For MPICH, the following environment variable is required:

```bash
export MSGPASS_USE_MPICH=1
```

For Intel MPI, the following commands are required:

```bash
source /opt/intel/oneapi/setvars.sh
export MSGPASS_USE_INTEL_MPI=1
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

## <a name="cargo"></a> Setting Cargo.toml

[![Crates.io](https://img.shields.io/crates/v/msgpass.svg)](https://crates.io/crates/msgpass)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
msgpass = "*"
```

## <a name="examples"></a> Examples

See also:

* [examples](https://github.com/cpmech/msgpass/tree/main/examples)

The example below (available in the `examples` directory) will send an array from ROOT to all the other processors.

```rust
use msgpass::*;

fn main() -> Result<(), StrError> {
    mpi_init()?;

    let mut comm = Communicator::new()?;
    let rank = comm.rank()?;
    let size = comm.size()?;

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
- [x] Wrap more MPI functions
    - [x] Implement send/receive
    - [x] Implement reduce/allreduce
    - [x] Implement scatter/gather/allgather
- [x] Handle complex numbers
