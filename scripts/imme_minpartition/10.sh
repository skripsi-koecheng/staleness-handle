#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# echo "============================================================"
# echo "Imme 1/6: Async Immediate + LoRA, Unweighted, Slow Dominant (Non-IID, Additive Delay)"
# echo "============================================================"
# ./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='slow_dominant' staleness-weighting-enabled=false wandb-run-name='63-noniid-imme-unweighted-slow-additive'"

# echo "============================================================"
# echo "Imme 2/6: Async Immediate + LoRA, Polynomial (α=0.25), Slow Dominant (Non-IID, Additive Delay)"
# echo "============================================================"
# ./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='slow_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 wandb-run-name='63-noniid-imme-weighted-poly025-slow-additive'"

echo "============================================================"
echo "Imme 3/6: Async Immediate + LoRA, Unweighted, Balanced (Non-IID, Additive Delay)"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='balanced' staleness-weighting-enabled=false wandb-run-name='10-imme-unweighted-balanced-additive'"

echo "============================================================"
echo "Imme 4/6: Async Immediate + LoRA, Polynomial (α=0.25), Balanced (Non-IID, Additive Delay)"
echo "============================================================"
./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 wandb-run-name='10-imme-weighted-poly025-balanced-additive'"

# echo "============================================================"
# echo "Imme 5/6: Async Immediate + LoRA, Unweighted, Fast Dominant (Non-IID, Additive Delay)"
# echo "============================================================"
# ./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=false wandb-run-name='63-noniid-imme-unweighted-fast-additive'"

# echo "============================================================"
# echo "Imme 6/6: Async Immediate + LoRA, Polynomial (α=0.25), Fast Dominant (Non-IID, Additive Delay)"
# echo "============================================================"
# ./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.5 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=16 min-evaluate-nodes=8 min-available-nodes=16 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=24 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='additive' fast-delay-seconds=0.0 medium-delay-seconds=20.0 slow-delay-seconds=40.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 wandb-run-name='63-noniid-imme-weighted-poly025-fast-additive'"

# echo "============================================================"
# echo "All Non-IID immediate runs completed (6/6)."
# echo "============================================================"
