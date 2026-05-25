#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "PoC Run 1/6: SyncFL + LoRA, 1 client, 1 round"
echo "============================================================"
./run.sh sync "num-server-rounds=1 fraction-train=0.005 fraction-evaluate=0.005 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=1 min-evaluate-nodes=1 min-available-nodes=1 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' straggler-enabled=false train-timeout-seconds=600.0 wandb-run-name='poc-sync-lora-1client-1round'"

echo "============================================================"
echo "PoC Run 2/6: SyncFL + Full Fine-Tuning, 1 client, 1 round"
echo "============================================================"
./run.sh sync "num-server-rounds=1 fraction-train=0.005 fraction-evaluate=0.005 local-epochs=1 learning-rate=0.00005 batch-size=4 min-train-nodes=1 min-evaluate-nodes=1 min-available-nodes=1 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=false sync-optimizer='fedavg' straggler-enabled=false train-timeout-seconds=600.0 wandb-run-name='poc-sync-fullft-1client-1round'"

echo "============================================================"
echo "PoC Run 3/6: Async Immediate + LoRA, 1 client, 1 round"
echo "============================================================"
./run.sh async "num-server-rounds=1 fraction-train=0.005 fraction-evaluate=0.005 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=1 min-evaluate-nodes=1 min-available-nodes=1 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=1 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=false staleness-weighting-enabled=false wandb-run-name='poc-immediate-lora-1client-1round'"

echo "============================================================"
echo "PoC Run 4/6: Async Immediate + Full Fine-Tuning, 1 client, 1 round"
echo "============================================================"
./run.sh async "num-server-rounds=1 fraction-train=0.005 fraction-evaluate=0.005 local-epochs=1 learning-rate=0.00005 batch-size=4 min-train-nodes=1 min-evaluate-nodes=1 min-available-nodes=1 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=false async-strategy='immediate' async-max-in-flight=1 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=false staleness-weighting-enabled=false wandb-run-name='poc-immediate-fullft-1client-1round'"

echo "============================================================"
echo "PoC Run 5/6: Async Buffered + LoRA, 1 client, 1 round"
echo "============================================================"
./run.sh async "num-server-rounds=1 fraction-train=0.005 fraction-evaluate=0.005 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=1 min-evaluate-nodes=1 min-available-nodes=1 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=1 async-max-in-flight=1 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=false staleness-weighting-enabled=false wandb-run-name='poc-buffered-lora-1client-1round'"

echo "============================================================"
echo "PoC Run 6/6: Async Buffered + Full Fine-Tuning, 1 client, 1 round"
echo "============================================================"
./run.sh async "num-server-rounds=1 fraction-train=0.005 fraction-evaluate=0.005 local-epochs=1 learning-rate=0.00005 batch-size=4 min-train-nodes=1 min-evaluate-nodes=1 min-available-nodes=1 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=false async-strategy='buffered' async-buffer-size=1 async-max-in-flight=1 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=false staleness-weighting-enabled=false wandb-run-name='poc-buffered-fullft-1client-1round'"

echo "============================================================"
echo "All PoC runs completed."
echo "============================================================"
