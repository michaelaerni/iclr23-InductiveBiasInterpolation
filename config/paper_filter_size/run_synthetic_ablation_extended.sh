#!/usr/bin/env bash

EXPERIMENT_NAME="filter_size"
RUN_NAME="synthetic"
SUBSET="ablation_extended"

# Determine common paths
REPOSITORY_DIR="$(realpath $(dirname $BASH_SOURCE)/../../)"
MODULE_DIR="${REPOSITORY_DIR}/src"
DATA_CACHE_DIR="/tmp/tradeoffdata/synthetic"

# Append Python module to python path in order to more cleanly call the experiment
export PYTHONPATH="${MODULE_DIR}:${PYTHONPATH}"

RUN_WRAPPER="${REPOSITORY_DIR}/run_wrapper.sh"
BINDINGS_FILE="${REPOSITORY_DIR}/config/paper_${EXPERIMENT_NAME}/${RUN_NAME}_${SUBSET}_bindings.txt"
num_bindings=$(wc -l < "${BINDINGS_FILE}" | xargs)

echo "Performing ${num_bindings} runs"
pushd "${REPOSITORY_DIR}"
for binding_idx in $(seq 1 $num_bindings); do
  "${RUN_WRAPPER}" ${binding_idx} "${BINDINGS_FILE}" \
    python -m filter_size \
      --experiment "paper_${EXPERIMENT_NAME}_${RUN_NAME}" \
      --run "${SUBSET}_${binding_idx}" \
      --config "${REPOSITORY_DIR}/config/paper_${EXPERIMENT_NAME}/${RUN_NAME}.gin" \
      --tag "${SUBSET}"\
      --bindings "cache_dir='${DATA_CACHE_DIR}'"
done
popd
