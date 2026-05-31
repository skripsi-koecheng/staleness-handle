#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# echo "============================================================"
# echo "Buff 1/6: Async Buffered + LoRA, Unweighted, Balanced (Additive)"
# echo "============================================================"
# ./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.0005 batch-size=32 dirichlet-alpha=1.0 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=8 async-max-in-flight=16 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=10.0 slow-delay-seconds=30.0 straggler-scenario='balanced' staleness-weighting-enabled=false staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.9 wandb-run-name='buff-unweighted-balanced-a1-b8-e09'"

# echo "============================================================"
# echo "Buff 2/6: Async Buffered + LoRA, FedStaleWeight, Balanced (Additive)"
# echo "============================================================"
# ./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.0005 batch-size=32 dirichlet-alpha=1.0 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=8 async-max-in-flight=16 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=10.0 slow-delay-seconds=30.0 straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.9 wandb-run-name='buff-weighted-fsw-balanced-a1-b8-e09'"

echo "============================================================"
echo "Buff 3/6: Async Buffered + LoRA, Unweighted, Slow Dominant (Additive)"
echo "============================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.0005 batch-size=32 dirichlet-alpha=1.0 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=8 async-max-in-flight=16 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=10.0 slow-delay-seconds=30.0 straggler-scenario='slow_dominant' staleness-weighting-enabled=false staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.9 wandb-run-name='buff-unweighted-slow-a1-b8-e09'"

echo "============================================================"
echo "Buff 4/6: Async Buffered + LoRA, FedStaleWeight, Slow Dominant (Additive)"
echo "============================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.0005 batch-size=32 dirichlet-alpha=1.0 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=8 async-max-in-flight=16 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=10.0 slow-delay-seconds=30.0 straggler-scenario='slow_dominant' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.9 wandb-run-name='buff-weighted-fsw-slow-a1-b8-e09'"

echo "============================================================"
echo "Buff 5/6: Async Buffered + LoRA, Unweighted, Fast Dominant (Additive)"
echo "============================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.0005 batch-size=32 dirichlet-alpha=1.0 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=8 async-max-in-flight=16 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=10.0 slow-delay-seconds=30.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=false staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.9 wandb-run-name='buff-unweighted-fast-a1-b8-e09'"

echo "============================================================"
echo "Buff 6/6: Async Buffered + LoRA, FedStaleWeight, Fast Dominant (Additive)"
echo "============================================================"
./run.sh async "num-server-rounds=50 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.0005 batch-size=32 dirichlet-alpha=1.0 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=8 async-max-in-flight=16 async-evaluate-interval=1 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=10.0 slow-delay-seconds=30.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.9 wandb-run-name='buff-weighted-fsw-fast-a1-b8-e09'"

echo "============================================================"
echo "All buffered alpha1 buffer8 runs completed (6/6)."
echo "============================================================"
