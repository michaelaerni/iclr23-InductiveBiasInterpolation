#!/usr/bin/env bash

RUN_IDX=$1
BINDINGS_FILE=$2
shift 2

echo "Running binding ${RUN_IDX}"
bindings=$(awk -v jindex=$RUN_IDX 'NR==jindex' "${BINDINGS_FILE}")
"$@" $bindings
