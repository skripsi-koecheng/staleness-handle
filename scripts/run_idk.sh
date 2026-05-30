#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Buff 1/4: Async Buffered + LoRA, Unweighted, Balanced (Measured)"
echo "============================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=4 async-max-in-flight=24 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='balanced' staleness-weighting-enabled=false wandb-run-name='l4-buff-unweighted-balanced-measured'"

echo "============================================================"
echo "Buff 2/4: Async Buffered + LoRA, FedStaleWeight, Balanced (Measured)"
echo "============================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=4 async-max-in-flight=24 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='l4-buff-fedstale-balanced-measured'"

echo "============================================================"
echo "Buff 3/4: Async Buffered + LoRA, Unweighted, Balanced (Additive Delay)"
echo "============================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=4 async-max-in-flight=24 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=10.0 slow-delay-seconds=20.0 straggler-scenario='balanced' staleness-weighting-enabled=false wandb-run-name='l4-buff-unweighted-balanced-additive'"

echo "============================================================"
echo "Buff 4/4: Async Buffered + LoRA, FedStaleWeight, Balanced (Additive Delay)"
echo "============================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=4 async-max-in-flight=24 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=10.0 slow-delay-seconds=20.0 straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='l4-buff-fedstale-balanced-additive'"

echo "============================================================"
echo "All L4 IDK buffered runs completed (4/4)."
echo "============================================================"
