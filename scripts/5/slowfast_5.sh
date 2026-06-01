#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

run_case() {
  local title="$1"
  local command="$2"

  echo "============================================================"
  echo "$title"
  echo "============================================================"
  ./run.sh async "$command"
}

base_config_alpha05="num-server-rounds=400 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=10.0 slow-delay-seconds=30.0"

run_case "Imme alpha=0.5 | Slow Dominant | Unweighted" \
  "$base_config_alpha05 straggler-scenario='slow_dominant' staleness-weighting-enabled=false dirichlet-alpha=0.5 wandb-run-name='imme-alpha05-slow-unweighted'"

run_case "Imme alpha=0.5 | Fast Dominant | Unweighted" \
  "$base_config_alpha05 straggler-scenario='fast_dominant' staleness-weighting-enabled=false dirichlet-alpha=0.5 wandb-run-name='imme-alpha05-fast-unweighted'"
