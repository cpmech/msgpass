#!/bin/bash

export CARGO_TARGET_DIR="/tmp/msgpass"

EXAMPLES="$CARGO_TARGET_DIR/debug/examples"

cargo build --example hello

mpirun -np 4 $EXAMPLES/hello
