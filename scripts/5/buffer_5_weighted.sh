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

base_config="num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=8 async-max-in-flight=16 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=10.0 slow-delay-seconds=20.0 dirichlet-alpha=0.5"

run_case "Buff alpha=0.5 | Balanced | FedStaleWeight" \
  "$base_config straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.9 wandb-run-name='buff-alpha05-balanced-fsw'"

run_case "Buff alpha=0.5 | Slow Dominant | FedStaleWeight" \
  "$base_config straggler-scenario='slow_dominant' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.9 wandb-run-name='buff-alpha05-slow-fsw'"

run_case "Buff alpha=0.5 | Fast Dominant | FedStaleWeight" \
  "$base_config straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.9 wandb-run-name='buff-alpha05-fast-fsw'"
