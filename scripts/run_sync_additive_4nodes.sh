#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Sync 1/3: SyncFL + LoRA, FedAvg, Balanced (Additive)"
echo "============================================================"
./run.sh sync "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.0005 batch-size=32 dirichlet-alpha=0.5 min-train-nodes=4 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' train-timeout-seconds=600.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='balanced' wandb-run-name='sync-fedavg-balanced-additive-mt4/05'"

echo "============================================================"
echo "Sync 2/3: SyncFL + LoRA, FedAvg, Slow Dominant (Additive)"
echo "============================================================"
./run.sh sync "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.0005 batch-size=32 dirichlet-alpha=0.5 min-train-nodes=4 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' train-timeout-seconds=600.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='slow_dominant' wandb-run-name='sync-fedavg-slow-additive-mt4/05'"

echo "============================================================"
echo "Sync 3/3: SyncFL + LoRA, FedAvg, Fast Dominant (Additive)"
echo "============================================================"
./run.sh sync "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.0005 batch-size=32 dirichlet-alpha=0.5 min-train-nodes=4 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' train-timeout-seconds=600.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='fast_dominant' wandb-run-name='sync-fedavg-fast-additive-mt4/05'"
