#!/usr/bin/env bash
# Spark 1 client, controlled through Zima tmux. Pure decode metric is tg.
set -euo pipefail
label=${1:?label required}
[[ "$label" =~ ^[A-Za-z0-9._-]+$ ]]
root=/home/mj-kang/Dev/benchmark/llama-benchy/results/k275-speed-20260908/$label
model=/home/mj-kang/Dev/models/GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-20260907T210500Z
test ! -e "$root"
mkdir -p "$root"
cd /home/mj-kang/Dev/benchmark/llama-benchy
git rev-parse HEAD > "$root/benchy-commit.txt"
git diff --stat > "$root/benchy-diff-stat.txt"
exec .venv/bin/llama-benchy --base-url http://127.0.0.1:8893/v1 \
 --model GLM-5.3-K3-TP3-CANDIDATE --tokenizer "$model" \
 --pp 2048 --tg 256 --exact-tg --depth 0 --runs 3 --concurrency 1 \
 --no-cache --latency-mode generation --exit-on-first-fail \
 --save-result "$root/result.json" --format json --emit-progress "$root/progress.jsonl" \
 > "$root/llama-benchy.log" 2>&1
