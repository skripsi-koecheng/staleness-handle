#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Imme 2/3: Async Immediate + LoRA, Polynomial (α=0.5), Balanced"
echo "============================================================"
./run.sh async "num-server-rounds=100 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.5 wandb-run-name='rtx-imme-weighted-poly05-balanced'"

echo "============================================================"
echo "Imme 1/3: Async Immediate + LoRA, Unweighted, Balanced"
echo "============================================================"
./run.sh async "num-server-rounds=100 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='balanced' staleness-weighting-enabled=false wandb-run-name='rtx-imme-unweighted-balanced'"

echo "============================================================"
echo "Imme 3/3: Async Immediate + LoRA, Polynomial (α=1.0), Balanced"
echo "============================================================"
./run.sh async "num-server-rounds=100 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=1.0 wandb-run-name='rtx-imme-weighted-poly10-balanced'"