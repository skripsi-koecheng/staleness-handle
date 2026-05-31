#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_case() {
  local title="$1"
  local command="$2"

  echo "============================================================"
  echo "$title"
  echo "============================================================"
  ./run.sh async "$command"
}

base_config_alpha025="num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=30.0"
base_config_alpha1="num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=20.0"

run_case "Imme alpha=0.25 | Fast Dominant | Unweighted" \
  "$base_config_alpha025 straggler-scenario='fast_dominant' staleness-weighting-enabled=false dirichlet-alpha=0.25 wandb-run-name='imme-alpha025-fast-unweighted'"

run_case "Imme alpha=0.25 | Fast Dominant | Weighted" \
  "$base_config_alpha025 straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 dirichlet-alpha=0.25 wandb-run-name='imme-alpha025-fast-weighted'"

run_case "Imme alpha=0.25 | Slow Dominant | Unweighted" \
  "$base_config_alpha025 straggler-scenario='slow_dominant' staleness-weighting-enabled=false dirichlet-alpha=0.25 wandb-run-name='imme-alpha025-slow-unweighted'"

run_case "Imme alpha=0.25 | Slow Dominant | Weighted" \
  "$base_config_alpha025 straggler-scenario='slow_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 dirichlet-alpha=0.25 wandb-run-name='imme-alpha025-slow-weighted'"

run_case "Imme alpha=1.0 | Fast Dominant | Unweighted" \
  "$base_config_alpha1 straggler-scenario='fast_dominant' staleness-weighting-enabled=false dirichlet-alpha=1.0 wandb-run-name='imme-alpha1-fast-unweighted'"

run_case "Imme alpha=1.0 | Fast Dominant | Weighted" \
  "$base_config_alpha1 straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 dirichlet-alpha=1.0 wandb-run-name='imme-alpha1-fast-weighted'"

run_case "Imme alpha=1.0 | Slow Dominant | Unweighted" \
  "$base_config_alpha1 straggler-scenario='slow_dominant' staleness-weighting-enabled=false dirichlet-alpha=1.0 wandb-run-name='imme-alpha1-slow-unweighted'"

run_case "Imme alpha=1.0 | Slow Dominant | Weighted" \
  "$base_config_alpha1 straggler-scenario='slow_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 dirichlet-alpha=1.0 wandb-run-name='imme-alpha1-slow-weighted'"
