#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Buff 1/2: Async Buffered + LoRA, Unweighted, Balanced (Non-IID)"
echo "============================================================"
./run.sh async "num-server-rounds=50 target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 learning-rate=0.0003 batch-size=32 max-length=256 min-train-nodes=16 min-evaluate-nodes=16 min-available-nodes=16 train-timeout-seconds=600.0 async-max-in-flight=24 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0 medium-delay-seconds=20 slow-delay-seconds=40 wandb-run-name='rtx-buff-unweighted-balanced-noniid-v5'"

echo "============================================================"
echo "Buff 2/2: Async Buffered + LoRA, FedStaleWeight, Balanced (Non-IID)"
echo "============================================================"
./run.sh async "num-server-rounds=50 target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 learning-rate=0.0003 batch-size=32 max-length=256 min-train-nodes=16 min-evaluate-nodes=16 min-available-nodes=16 train-timeout-seconds=600.0 async-max-in-flight=24 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0 medium-delay-seconds=20 slow-delay-seconds=40 staleness-weighting-enabled=true wandb-run-name='rtx-buff-weighted-fsw-balanced-noniid-v5'"

echo "============================================================"
echo "All buffered Non-IID v5 runs completed (2/2)."
echo "============================================================"