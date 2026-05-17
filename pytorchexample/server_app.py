from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from pytorchexample.synchronous import (
    SynchronousFedAdagradStrategy,
    SynchronousFedAvgStrategy,
)
from pytorchexample.task import (
    DistilBertAgNewsClassifier,
    global_evaluate,
    set_global_seed,
    set_use_lora,
    GLOBAL_MODEL_SEED,
)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    stop_mode: str = str(context.run_config.get("stop-mode", "num_rounds"))
    target_accuracy: float = float(
        context.run_config.get("target-accuracy", 0.9))
    lr: float = context.run_config["learning-rate"]
    weight_decay: float = context.run_config.get("weight-decay", 0.01)
    warmup_ratio: float = context.run_config.get("warmup-ratio", 0.1)
    max_grad_norm: float = context.run_config.get("max-grad-norm", 1.0)
    fraction_train: float = context.run_config["fraction-train"]
    lr_decay_interval: int = int(
        context.run_config.get("lr-decay-interval", 20))
    lr_decay_factor: float = float(
        context.run_config.get("lr-decay-factor", 0.9))
    min_learning_rate: float = float(
        context.run_config.get("min-learning-rate", 1e-4))
    min_train_nodes: int = int(context.run_config.get("min-train-nodes", 8))
    min_evaluate_nodes: int = int(
        context.run_config.get("min-evaluate-nodes", 8))
    min_available_nodes: int = int(
        context.run_config.get("min-available-nodes", 8))
    use_lora: bool = bool(context.run_config.get("use-lora", True))

    sync_optimizer: str = str(
        context.run_config.get("sync-optimizer", "fedadagrad")
    ).lower()

    # Load global model (seeded for deterministic initialization)
    set_global_seed(GLOBAL_MODEL_SEED)
    set_use_lora(use_lora)
    global_model = DistilBertAgNewsClassifier(use_lora=use_lora)
    arrays = ArrayRecord(global_model.get_federated_state_dict())

    if sync_optimizer in {"fedavg", "avg"}:
        strategy_cls = SynchronousFedAvgStrategy
    elif sync_optimizer in {"fedadagrad", "adagrad"}:
        strategy_cls = SynchronousFedAdagradStrategy
    else:
        raise ValueError(
            "Invalid sync-optimizer. Use 'fedavg' or 'fedadagrad'."
        )

    # Initialize sync strategy
    strategy = strategy_cls(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        lr_decay_interval=lr_decay_interval,
        lr_decay_factor=lr_decay_factor,
        min_learning_rate=min_learning_rate,
        min_available_nodes=min_available_nodes,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=min_evaluate_nodes,
    )

    # Start strategy, run FedAvg for `num_rounds`
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "max_grad_norm": max_grad_norm,
            }
        ),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
        stop_mode=stop_mode,
        target_accuracy=target_accuracy,
    )
