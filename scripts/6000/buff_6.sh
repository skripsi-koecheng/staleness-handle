#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# echo "============================================================"
# echo "Buff 1/6: Ameowsync Buffered + LoRA, Unweighted, Fast Dominant (Additive)"
# echo "============================================================"
# ./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=16 min-train-nodes=16 min-evaluate-nodes=16 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=24 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=false wandb-run-name='62-noniid-buff-6-fixed-unweighted-fast'"

echo "============================================================"
echo "Buff 2/6: Async Buffered + LoRA, FedStaleWeight, Fast Dominant (Additive)"
echo "============================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=16 min-train-nodes=16 min-evaluate-nodes=16 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=24 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='62-noniid-buff-6-fixed-fedstale-fast'"

echo "============================================================"
echo "Buff 1/6: Async Buffered + LoRA, Unweighted, Fast Dominant (Additive)"
echo "============================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=16 min-train-nodes=16 min-evaluate-nodes=16 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=8 async-max-in-flight=24 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=false wandb-run-name='62-noniid-buff-8-fixed-unweighted-fast'"

echo "============================================================"
echo "Buff 2/6: Async Buffered + LoRA, FedStaleWeight, Fast Dominant (Additive)"
echo "============================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=16 min-train-nodes=16 min-evaluate-nodes=16 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=8 async-max-in-flight=24 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='62-noniid-buff-8-fixed-fedstale-fast'"

echo "========================================================"
echo "Buff 1/6: Async Buffered + LoRA, Unweighted, Fast Dominant (Additive)"
echo "========================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=16 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=24 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=false wandb-run-name='62-noniid-buff-6/32-fixed-unweighted-fast'"

echo "========================================================"
echo "Buff 2/6: Async Buffered + LoRA, FedStaleWeight, Fast Dominant (Additive)"
echo "========================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=16 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=24 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='62-noniid-buff-6/32-fixed-fedstale-fast'"