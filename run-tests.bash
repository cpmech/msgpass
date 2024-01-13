#!/bin/bash

set -e

NP=4
if [[ "$CI" == "true" ]]; then
    NP=2
fi

export CARGO_TARGET_DIR="/tmp/msgpass"

EXAMPLES="/tmp/msgpass/debug/examples"

rm -rf $EXAMPLES
cargo build --examples

for test in examples/test_*.rs; do
    filename="$(basename "$test")"
    filekey="${filename%%.*}"

    echo
    echo "### $filekey ######################################################"

    mpirun -np $NP $EXAMPLES/$filekey
done
