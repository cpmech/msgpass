#!/bin/bash

set -e

# the first argument is the feature: "", "intel", or "mpich"
FEATURE=${1:-""}

NP=4

export CARGO_TARGET_DIR="/tmp/msgpass"

EXAMPLES="/tmp/msgpass/debug/examples"

rm -rf $EXAMPLES
if [ "${FEATURE}" = "" ]; then
    cargo build --examples
else
    cargo build --examples --features $FEATURE
fi

for example in examples/ex_*.rs; do
    filename="$(basename "$example")"
    filekey="${filename%%.*}"

    echo
    echo "### $filekey ######################################################"

    mpiexec -np $NP $EXAMPLES/$filekey
done
