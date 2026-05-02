import io
import time
from contextlib import suppress
from logging import INFO
from typing import Callable, Iterable, Optional

import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

from pytorchexample.task import get_top1_test_accuracy

PROJECT_NAME = "flower-async-staleness"


class SynchronousStrategy(FedAdagrad):

    def __init__(
        self,
        *args,
        lr_decay_interval: int = 20,
        lr_decay_factor: float = 0.9,
        min_learning_rate: float = 1e-4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr_decay_interval = max(1, lr_decay_interval)
        self.lr_decay_factor = min(1.0, max(0.0, lr_decay_factor))
        self.min_learning_rate = max(0.0, min_learning_rate)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
        if (
            server_round > 0
            and server_round % self.lr_decay_interval == 0
            and self.lr_decay_factor < 1.0
        ):
            current_lr = float(config.get("lr", 0.0))
            if current_lr > self.min_learning_rate:
                new_lr = max(
                    self.min_learning_rate,
                    current_lr * self.lr_decay_factor,
                )
                if new_lr < current_lr:
                    config["lr"] = new_lr
                    log(
                        INFO,
                        "[SYNC] LR decayed at round %s: %.6f -> %.6f",
                        server_round,
                        current_lr,
                        new_lr,
                    )
        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
        stop_mode: str = "num_rounds",
        target_accuracy: float = 0.9,
    ) -> Result:
        """Execute the federated learning strategy while logging results to W&B."""

        # Init W&B
        wandb.init(project=PROJECT_NAME)

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()
        target_mode = str(stop_mode).strip().lower() in {
            "target_accuracy",
            "target-accuracy",
            "target",
        }

        def log_target_metrics(step: int, client_trips: int) -> None:
            wall_clock_seconds = time.time() - t_start
            log(
                INFO,
                "[TARGET] Top-1 Test Accuracy reached: target=%.4f | client_trips=%d | wall_clock=%.2fs",
                target_accuracy,
                client_trips,
                wall_clock_seconds,
            )
            wandb.log(
                {
                    "number_of_client_trips_to_target_accuracy": client_trips,
                    "wall_clock_time_to_target_accuracy": wall_clock_seconds,
                },
                step=step,
            )

        try:
            t_start = time.time()
            client_trips_done = 0
            # Evaluate starting global parameters
            if evaluate_fn:
                res = evaluate_fn(0, initial_arrays)
                log(INFO, "Initial global evaluation results: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[0] = res
                    if target_mode:
                        accuracy = get_top1_test_accuracy(res)
                        if accuracy is not None and accuracy >= target_accuracy:
                            log_target_metrics(0, client_trips_done)
                            return result

            arrays = initial_arrays

            for current_round in range(1, num_rounds + 1):
                log(INFO, "")
                log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

                # -----------------------------------------------------------------
                # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
                # -----------------------------------------------------------------

                # Call strategy to configure training round
                # Send messages and wait for replies
                train_replies = list(grid.send_and_receive(
                    messages=self.configure_train(
                        current_round,
                        arrays,
                        train_config,
                        grid,
                    ),
                    timeout=timeout,
                ))

                # Aggregate train
                agg_arrays, agg_train_metrics = self.aggregate_train(
                    current_round,
                    train_replies,
                )

                # Log training metrics and append to history
                if agg_arrays is not None:
                    result.arrays = agg_arrays
                    arrays = agg_arrays
                if agg_train_metrics is not None:
                    log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                    result.train_metrics_clientapp[current_round] = agg_train_metrics
                    # Log to W&B
                    wandb.log(dict(agg_train_metrics), step=current_round)
                    client_trips_done += len(train_replies)

                # -----------------------------------------------------------------
                # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
                # -----------------------------------------------------------------

                # Call strategy to configure evaluation round
                # Send messages and wait for replies
                evaluate_replies = grid.send_and_receive(
                    messages=self.configure_evaluate(
                        current_round,
                        arrays,
                        evaluate_config,
                        grid,
                    ),
                    timeout=timeout,
                )

                # Aggregate evaluate
                agg_evaluate_metrics = self.aggregate_evaluate(
                    current_round,
                    evaluate_replies,
                )

                # Log training metrics and append to history
                if agg_evaluate_metrics is not None:
                    log(INFO, "\t└──> Aggregated MetricRecord: %s",
                        agg_evaluate_metrics)
                    result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                    # Log to W&B
                    wandb.log(dict(agg_evaluate_metrics), step=current_round)
                # -----------------------------------------------------------------
                # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
                # -----------------------------------------------------------------

                # Centralized evaluation
                if evaluate_fn:
                    log(INFO, "Global evaluation")
                    res = evaluate_fn(current_round, arrays)
                    log(INFO, "\t└──> MetricRecord: %s", res)
                    if res is not None:
                        result.evaluate_metrics_serverapp[current_round] = res
                        # Log to W&B
                        wandb.log(dict(res), step=current_round)
                        if target_mode:
                            accuracy = get_top1_test_accuracy(res)
                            if accuracy is not None and accuracy >= target_accuracy:
                                log_target_metrics(
                                    current_round, client_trips_done)
                                return result

            log(INFO, "")
            log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
            log(INFO, "")
            log(INFO, "Final results:")
            log(INFO, "")
            for line in io.StringIO(str(result)):
                log(INFO, "\t%s", line.strip("\n"))
            log(INFO, "")

            return result
        finally:
            with suppress(Exception):
                wandb.finish()
