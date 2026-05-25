#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "Run 10/15: Async Immediate + LoRA, Unweighted, Balanced"
echo "============================================================"
./run.sh async "num-server-rounds=100 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='balanced' staleness-weighting-enabled=false wandb-run-name='immediate-lora-unweighted-balanced'"

echo "============================================================"
echo "Run 11/15: Async Immediate + LoRA, Unweighted, Slow Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=100 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='slow_dominant' staleness-weighting-enabled=false wandb-run-name='immediate-lora-unweighted-slow'"

echo "============================================================"
echo "Run 12/15: Async Immediate + LoRA, Unweighted, Fast Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=100 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='fast_dominant' staleness-weighting-enabled=false wandb-run-name='immediate-lora-unweighted-fast'"

echo "============================================================"
echo "Run 13/15: Async Immediate + LoRA, Polynomial Weighted, Balanced"
echo "============================================================"
./run.sh async "num-server-rounds=100 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.5 wandb-run-name='immediate-lora-weighted-balanced'"

echo "============================================================"
echo "Run 14/15: Async Immediate + LoRA, Polynomial Weighted, Slow Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=100 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='slow_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.5 wandb-run-name='immediate-lora-weighted-slow'"

echo "============================================================"
echo "Run 15/15: Async Immediate + LoRA, Polynomial Weighted, Fast Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=100 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=12 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.5 wandb-run-name='immediate-lora-weighted-fast'"

echo "============================================================"
echo "All Async Immediate + LoRA runs completed."
echo "============================================================"
