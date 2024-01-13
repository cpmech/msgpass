#!/bin/bash

set -e

NP=4
if [[ "$CI" == "true" ]]; then
    NP=1
fi

export CARGO_TARGET_DIR="/tmp/msgpass"

EXAMPLES="/tmp/msgpass/debug/examples"

TEST="test_reduce"

rm -rf $EXAMPLES
cargo build --example $TEST

echo
echo
echo "### $TEST ######################################################"

mpirun -np $NP $EXAMPLES/$TEST

