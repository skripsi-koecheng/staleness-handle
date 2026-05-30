#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ============================================================
# ASYNC IMMEDIATE — 6 runs
# ============================================================

# echo "============================================================"
# echo "Imme 1/6: Async Immediate + LoRA, Unweighted, Balanced"
# echo "============================================================"
# ./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='balanced' staleness-weighting-enabled=false wandb-run-name='rtx-imme-unweighted-balanced'"

# echo "============================================================"
# echo "Imme 2/6: Async Immediate + LoRA, Unweighted, Slow Dominant"
# echo "============================================================"
# ./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='slow_dominant' staleness-weighting-enabled=false wandb-run-name='rtx-imme-unweighted-slow'"

echo "============================================================"
echo "Imme 3/6: Async Immediate + LoRA, Unweighted, Fast Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=false wandb-run-name='rtx-imme-unweighted-fast'"

echo "============================================================"
echo "Imme 4/6: Async Immediate + LoRA, Polynomial Weighted (alpha=0.25), Balanced"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 wandb-run-name='rtx-imme-weighted-poly025-balanced'"

echo "============================================================"
echo "Imme 4/6: Async Immediate + LoRA, Polynomial Weighted (alpha=0.5), Balanced"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.5 wandb-run-name='rtx-imme-weighted-poly05-balanced'"

echo "============================================================"
echo "Imme 5/6: Async Immediate + LoRA, Polynomial Weighted (alpha=0.25), Slow Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='slow_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 wandb-run-name='rtx-imme-weighted-poly025-slow'"

echo "============================================================"
echo "Imme 6/6: Async Immediate + LoRA, Polynomial Weighted (alpha=0.25), Fast Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 wandb-run-name='rtx-imme-weighted-poly025-fast'"

# ============================================================
# ASYNC BUFFERED — 6 runs
# ============================================================

echo "============================================================"
echo "Buff 1/6: Async Buffered + LoRA, Unweighted, Balanced"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=4 async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='balanced' staleness-weighting-enabled=false wandb-run-name='rtx-buff-unweighted-balanced'"

echo "============================================================"
echo "Buff 2/6: Async Buffered + LoRA, Unweighted, Slow Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=4 async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='slow_dominant' staleness-weighting-enabled=false wandb-run-name='rtx-buff-unweighted-slow'"

echo "============================================================"
echo "Buff 3/6: Async Buffered + LoRA, Unweighted, Fast Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=4 async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='fast_dominant' staleness-weighting-enabled=false wandb-run-name='rtx-buff-unweighted-fast'"

echo "============================================================"
echo "Buff 4/6: Async Buffered + LoRA, FedStaleWeight, Balanced"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=4 async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='rtx-buff-weighted-fsw-balanced'"

echo "============================================================"
echo "Buff 5/6: Async Buffered + LoRA, FedStaleWeight, Slow Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=4 async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='slow_dominant' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='rtx-buff-weighted-fsw-slow'"

echo "============================================================"
echo "Buff 6/6: Async Buffered + LoRA, FedStaleWeight, Fast Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=4 async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='rtx-buff-weighted-fsw-fast'"

# ============================================================
# SYNC FEDAVG — 3 runs
# ============================================================

echo "============================================================"
echo "Sync 1/3: SyncFL + LoRA, FedAvg, Balanced"
echo "============================================================"
./run.sh sync "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' train-timeout-seconds=600.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='balanced' wandb-run-name='rtx-sync-fedavg-balanced'"

echo "============================================================"
echo "Sync 2/3: SyncFL + LoRA, FedAvg, Slow Dominant"
echo "============================================================"
./run.sh sync "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' train-timeout-seconds=600.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='slow_dominant' wandb-run-name='rtx-sync-fedavg-slow'"

echo "============================================================"
echo "Sync 3/3: SyncFL + LoRA, FedAvg, Fast Dominant"
echo "============================================================"
./run.sh sync "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' train-timeout-seconds=600.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='fast_dominant' wandb-run-name='rtx-sync-fedavg-fast'"

echo "============================================================"
echo "Imme 6/6: Async Immediate + LoRA, Polynomial Weighted (alpha=0.25), Fast Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=68 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 wandb-run-name='rtx-imme-weighted-poly025-fast-batchsize68'"

echo "============================================================"
echo "Imme 5/6: Async Immediate + LoRA, Polynomial Weighted (alpha=0.25), Slow Dominant"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=68 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='slow_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 wandb-run-name='rtx-imme-weighted-poly025-slow-batchsize68'"

echo "============================================================"
echo "All RTX runs completed (15/15)."
echo "============================================================"
