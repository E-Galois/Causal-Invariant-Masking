#!/usr/bin/env bash
# Main CIM pipeline. Run from any directory; all paths resolve inside this repo.
# Select one visible GPU, e.g. CUDA_VISIBLE_DEVICES=0 bash commands.sh QwenVL VQAv2
# No arguments: both models on all seven benchmarks, sequentially.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PYTHONUNBUFFERED=1

models=(InternVL QwenVL)
benchmarks=(VQAv2 OKVQA AdVQA POPE_rand POPE_pop POPE_adv MME)
if (( $# >= 1 )); then models=("$1"); fi
if (( $# >= 2 )); then benchmarks=("$2"); fi
if (( $# > 2 )); then
  echo "Usage: bash commands.sh [InternVL|QwenVL] [benchmark]" >&2
  exit 2
fi
for mllm in "${models[@]}"; do
  for benchmark in "${benchmarks[@]}"; do
    # 1. Original answers, detected crops, sampling, and semantic clustering.
    python prepare_sd.py --mllm "$mllm" --benchmark "$benchmark" --prefix CIM
    # 2. EED teacher-forced expected and label embeddings.
    python prepare_eed.py --mllm "$mllm" --benchmark "$benchmark" --prefix CIM
    # 3. Pure metrics and fixed-weight SE blends; AUROC and original AURAC.
    python evaluate_uncertainty.py --mllm "$mllm" --benchmark "$benchmark" --prefix CIM --lambda 0.05
  done
done
