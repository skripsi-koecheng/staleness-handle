#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Run 16/21: Async Buffered + LoRA, Unweighted, Balanced"
echo "============================================================"
./run.sh async "num-server-rounds=100 target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='balanced' staleness-weighting-enabled=false wandb-run-name='buffered-lora-unweighted-balanced'"

echo "============================================================"
echo "Run 17/21: Async Buffered + LoRA, Unweighted, Slow Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=100 target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='slow_dominant' staleness-weighting-enabled=false wandb-run-name='buffered-lora-unweighted-slow'"

echo "============================================================"
echo "Run 18/21: Async Buffered + LoRA, Unweighted, Fast Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=100 target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='fast_dominant' staleness-weighting-enabled=false wandb-run-name='buffered-lora-unweighted-fast'"

echo "============================================================"
echo "Run 19/21: Async Buffered + LoRA, FedStaleWeight, Balanced"
echo "============================================================"
./run.sh async "num-server-rounds=100 target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='buffered-lora-weighted-balanced'"

echo "============================================================"
echo "Run 20/21: Async Buffered + LoRA, FedStaleWeight, Slow Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=100 target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='slow_dominant' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='buffered-lora-weighted-slow'"

echo "============================================================"
echo "Run 21/21: Async Buffered + LoRA, FedStaleWeight, Fast Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=100 target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='buffered-lora-weighted-fast'"

echo "============================================================"
echo "All Async Buffered + LoRA runs completed."
echo "============================================================"
