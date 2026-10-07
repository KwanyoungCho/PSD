#!/usr/bin/env bash
set -euo pipefail
# Invoke from any directory. Override MLSYS_PYTHON and CUDA_VISIBLE_DEVICES.
cd "$(dirname "${BASH_SOURCE[0]}")/../../ssd"
mlsys_python=${MLSYS_PYTHON:-/home/chokwans99/PSD/ssd/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SSD_CUDA_ARCH=${SSD_CUDA_ARCH:-8.9}
export SSD_HF_CACHE=${SSD_HF_CACHE:-/tmp}
export SSD_DATASET_DIR=${SSD_DATASET_DIR:-/tmp}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec "$mlsys_python" -m unittest \
  tests.test_greedy_verify tests.test_b_gt1_jit_subset \
  tests.test_b_gt1_m1 tests.test_b_gt1_m2 tests.test_b_gt1_m3 \
  tests.test_b_gt1_m4 tests.test_b_gt1_m6_verify_window \
  tests.test_batched_proxy tests.test_cached_prefill \
  tests.test_context_length_contract tests.test_model_pair_contract \
  tests.test_p2_tree_alloc tests.test_stochastic_verify \
  tests.test_output_accounting tests.test_greedy_sampler tests.test_colocated_norm \
  tests.test_packed_verify tests.test_greedy_tree tests.test_batched_tree_executor \
  tests.test_p1_dynamic_tree tests.test_p2_executor_parity tests.test_executor_premises \
  tests.test_tree_verify_planless tests.test_tree_host_topology \
  tests.test_batch_tree_serving
