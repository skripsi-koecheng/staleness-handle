#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Async Immediate + LoRA, Unweighted, Balanced, fixed5"
echo "============================================================"
./run.sh async "num-server-rounds=150 stop-mode='num_rounds' target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=20 async-evaluate-interval=2 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='balanced' staleness-weighting-enabled=false wandb-run-name='yelp-measured-balance-unweighted-imme'"

# echo "============================================================"
# echo "Async Immediate + LoRA, Unweighted, Slow Dominant, fixed5"
# echo "============================================================"
# ./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=16 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=20 async-evaluate-interval=5 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='fixed' baseline-time-seconds=5.0 straggler-scenario='slow_dominant' staleness-weighting-enabled=false wandb-run-name='immediate-lora-unweighted-slow-fixed5'"

# echo "============================================================"
# echo "Async Immediate + LoRA, Unweighted, Fast Dominant, fixed5"
# echo "============================================================"
# ./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=16 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=20 async-evaluate-interval=5 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='fixed' baseline-time-seconds=5.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=false wandb-run-name='immediate-lora-unweighted-fast-fixed5'"

echo "============================================================"
echo "Async Immediate + LoRA, Polynomial Weighted, Balanced, fixed5, alpha=0.25"
echo "============================================================"
./run.sh async "num-server-rounds=150 stop-mode='num_rounds' target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=20 async-evaluate-interval=2 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' baseline-time-seconds=5.0 straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 wandb-run-name='yelp-measured-balance-weighted-imme'"

# echo "============================================================"
# echo "Async Immediate + LoRA, Polynomial Weighted, Slow Dominant, fixed5, alpha=0.25"
# echo "============================================================"
# ./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=16 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=20 async-evaluate-interval=5 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='fixed' baseline-time-seconds=5.0 straggler-scenario='slow_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 wandb-run-name='immediate-lora-weighted-slow-fixed5-alpha025'"

# echo "============================================================"
# echo "Async Immediate + LoRA, Polynomial Weighted, Fast Dominant, fixed5, alpha=0.25"
# echo "============================================================"
# ./run.sh async "num-server-rounds=200 stop-mode='num_rounds' target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=16 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='immediate' async-max-in-flight=20 async-evaluate-interval=5 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='fixed' baseline-time-seconds=5.0 straggler-scenario='fast_dominant' staleness-weighting-enabled=true staleness-weighting-mode='polynomial' staleness-exponent=0.25 wandb-run-name='immediate-lora-weighted-fast-fixed5-alpha025'"

# echo "============================================================"
# echo "Run 16/21: Async Buffered + LoRA, Unweighted, Balanced"
# echo "============================================================"
# ./run.sh async "num-server-rounds=100 target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=20 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='balanced' staleness-weighting-enabled=false wandb-run-name='buffered-lora-unweighted-balanced'"

# echo "============================================================"
# echo "Run 19/21: Async Buffered + LoRA, FedStaleWeight, Balanced"
# echo "============================================================"
# ./run.sh async "num-server-rounds=100 target-accuracy=0.9 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.001 batch-size=32 min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true async-strategy='buffered' async-buffer-size=6 async-max-in-flight=20 async-evaluate-interval=4 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 straggler-enabled=true baseline-mode='measured' straggler-scenario='balanced' staleness-weighting-enabled=true staleness-weighting-mode='fedstaleweight' fedstaleweight-ema-beta=0.8 wandb-run-name='buffered-lora-weighted-balanced'"

# echo "============================================================"
# echo "All Async Immediate + LoRA fixed5 runs completed."
# echo "============================================================"