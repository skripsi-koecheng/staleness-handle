from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from pytorchexample.synchronous import SynchronousStrategy
from pytorchexample.task import DistilBertAgNewsClassifier, global_evaluate

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    fraction_train: float = context.run_config["fraction-train"]

    # Load global model
    global_model = DistilBertAgNewsClassifier()
    arrays = ArrayRecord(global_model.get_federated_state_dict())

    # Initialize FedAvg strategy
    strategy = SynchronousStrategy(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_available_nodes=10,
        min_train_nodes=10,
        min_evaluate_nodes=10,
    )

    # Start strategy, run FedAvg for `num_rounds`
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )
