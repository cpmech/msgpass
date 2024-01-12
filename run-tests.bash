#!/bin/bash

set -e

export CARGO_TARGET_DIR="/tmp/msgpass"

EXAMPLES="/tmp/msgpass/debug/examples"

rm -rf $EXAMPLES

for test in examples/test_*.rs; do
    filename="$(basename "$test")"
    filekey="${filename%%.*}"
    cargo build --example $filekey
    mpirun -np 4 $EXAMPLES/$filekey
done
