#!/bin/bash

set -e

NP=4

export CARGO_TARGET_DIR="/tmp/msgpass"

EXAMPLES="/tmp/msgpass/debug/examples"

rm -rf $EXAMPLES
cargo build --examples

for example in examples/ex_*.rs; do
    filename="$(basename "$example")"
    filekey="${filename%%.*}"

    echo
    echo "### $filekey ######################################################"

    mpiexec -np $NP $EXAMPLES/$filekey
done
