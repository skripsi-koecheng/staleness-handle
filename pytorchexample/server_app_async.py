"""pytorchexample async server app."""

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from pytorchexample.asynchronous import AsyncFedAvgStrategy
from pytorchexample.asynchronous_buffered import AsyncBufferedFedAvgStrategy
from pytorchexample.task import DistilBertAgNewsClassifier, global_evaluate

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for asynchronous ServerApp."""

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    weight_decay: float = context.run_config.get("weight-decay", 0.01)
    warmup_ratio: float = context.run_config.get("warmup-ratio", 0.1)
    max_grad_norm: float = context.run_config.get("max-grad-norm", 1.0)
    train_timeout: float = context.run_config.get(
        "train-timeout-seconds", 120.0)
    reply_poll_interval: float = context.run_config.get(
        "reply-poll-interval-seconds", 2.0)
    async_max_in_flight: int = context.run_config.get("async-max-in-flight", 4)
    async_evaluate_interval: int = context.run_config.get(
        "async-evaluate-interval", 4)
    async_buffer_size: int = context.run_config.get("async-buffer-size", 4)
    async_strategy: str = str(
        context.run_config.get("async-strategy", "immediate")
    ).lower()
    staleness_weighting_enabled: bool = bool(
        context.run_config.get("staleness-weighting-enabled", False)
    )
    staleness_exponent: float = float(
        context.run_config.get("staleness-exponent", 1.0)
    )

    global_model = DistilBertAgNewsClassifier()
    arrays = ArrayRecord(global_model.get_federated_state_dict())

    if async_strategy in {"buffered", "buffer"}:
        strategy = AsyncBufferedFedAvgStrategy(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            train_timeout=train_timeout,
            reply_poll_interval=reply_poll_interval,
            async_max_in_flight=async_max_in_flight,
            async_evaluate_interval=async_evaluate_interval,
            async_buffer_size=async_buffer_size,
            min_available_nodes=10,
            min_train_nodes=10,
            min_evaluate_nodes=10,
            staleness_weighting_enabled=staleness_weighting_enabled,
            staleness_exponent=staleness_exponent,
        )
    elif async_strategy in {"immediate", "plain"}:
        strategy = AsyncFedAvgStrategy(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            train_timeout=train_timeout,
            reply_poll_interval=reply_poll_interval,
            async_max_in_flight=async_max_in_flight,
            async_evaluate_interval=async_evaluate_interval,
            min_available_nodes=10,
            min_train_nodes=10,
            min_evaluate_nodes=10,
            staleness_weighting_enabled=staleness_weighting_enabled,
            staleness_exponent=staleness_exponent,
        )
    else:
        raise ValueError(
            "Invalid async-strategy. Use 'immediate' or 'buffered'."
        )

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
    )
