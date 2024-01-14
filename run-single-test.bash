#!/bin/bash

set -e

if [ "$#" -lt 1 ]; then
	echo
	echo "Usage:"
	echo "        $0 test_filekey"
	echo
	exit 1
fi

NP=4
if [[ "$CI" == "true" ]]; then
    NP=2
fi

export CARGO_TARGET_DIR="/tmp/msgpass"

EXAMPLES="/tmp/msgpass/debug/examples"

TEST=$1

rm -rf $EXAMPLES
cargo build --example $TEST

echo
echo
echo "### $TEST ######################################################"

mpiexec -np $NP $EXAMPLES/$TEST

