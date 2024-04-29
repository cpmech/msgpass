# Thin wrapper to a Message Passing Interface (MPI) <!-- omit from toc -->

[![Test on macOS](https://github.com/cpmech/msgpass/actions/workflows/test_on_macos.yml/badge.svg)](https://github.com/cpmech/msgpass/actions/workflows/test_on_macos.yml)
[![Test on Linux](https://github.com/cpmech/msgpass/actions/workflows/test_on_linux.yml/badge.svg)](https://github.com/cpmech/msgpass/actions/workflows/test_on_linux.yml)
[![Test on Linux with Intel MPI](https://github.com/cpmech/msgpass/actions/workflows/test_on_linux_intel_mpi.yml/badge.svg)](https://github.com/cpmech/msgpass/actions/workflows/test_on_linux_intel_mpi.yml)

[![documentation](https://img.shields.io/badge/msgpass-documentation-blue)](https://docs.rs/msgpass)

## Contents <!-- omit from toc -->

- [Introduction](#introduction)
- [Installation](#installation)
  - [Debian/Ubuntu Linux](#debianubuntu-linux)
  - [macOS](#macos)
  - [Setting Cargo.toml](#setting-cargotoml)
- [Examples](#examples)
- [Roadmap](#roadmap)



## Introduction

MsgPass (Message Passing) is a thin Rust wrapper to MPI. We consider a small subset of MPI functions. This subset will grow as our projects require more functionality. We implement (by hand) C functions that Rust can easily call using the FFI (in the `c_code` directory).

We try to test all functions as much as possible, but test coverage could be better. The tests must be called with `mpiexec`, thus it is easy to use the `run-tests.bash` script.

[![documentation](https://img.shields.io/badge/msgpass-documentation-blue)](https://docs.rs/msgpass)

**Note:** We can communicate strings by converting them to an array of bytes. For instance:

```rust
let mut bytes = vec![0_u8; MAX];
str_to_bytes(&mut bytes, "Hello World 😊");
comm.broadcast_bytes(0, &mut bytes)?;
```


## Installation

### Debian/Ubuntu Linux

First, install OpenMPI, MPICH, or Intel MPI. For instance,

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

The use the corresponding `feature`:

* `intel`: use Intel MPI 
* `mpich`: use MPICH
* (default): use OpenMPI

For Intel MPI, remember to call `setvars.sh` first:

```bash
source /opt/intel/oneapi/setvars.sh
```



### macOS

On **macOS**, install the following packages:


```bash
brew install llvm open-mpi
```

Also, export the following environment variable:

```bash
export echo TMPDIR=/tmp
```



### Setting Cargo.toml

[![Crates.io](https://img.shields.io/crates/v/msgpass.svg)](https://crates.io/crates/msgpass)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
msgpass = "*"
```

Or, considering the optional _features_:

```toml
[dependencies]
msgpass = { version = "*", features = ["intel"] }
```



## Examples

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



## Roadmap

- [x] Implement basic functionality
    - [x] Initialize and finalize
    - [x] Abort and barrier
- [x] Wrap more MPI functions
    - [x] Implement send/receive
    - [x] Implement reduce/allreduce
    - [x] Implement scatter/gather/allgather
- [x] Handle complex numbers
