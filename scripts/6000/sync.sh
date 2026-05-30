#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."


echo "Sync 1/3: SyncFL + LoRA, FedAvg, Balanced (Additive)"

./run.sh sync "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=16 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' train-timeout-seconds=600.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='balanced' wandb-run-name='62-sync-fedavg-balanced'"


echo "Sync 2/3: SyncFL + LoRA, FedAvg, Slow Dominant (Additive)"

./run.sh sync "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=16 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' train-timeout-seconds=600.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='slow_dominant' wandb-run-name='62-sync-fedavg-slow'"

echo "Sync 3/3: SyncFL + LoRA, FedAvg, Fast Dominant (Additive)"

./run.sh sync "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=16 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' train-timeout-seconds=600.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='fast_dominant' wandb-run-name='62-sync-fedavg-fast'"