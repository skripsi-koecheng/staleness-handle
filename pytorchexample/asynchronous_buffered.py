import io
import time
from contextlib import suppress
from logging import INFO
from typing import Callable, Iterable, Optional

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

from pytorchexample.staleness import polynomial_staleness_weight

PROJECT_NAME = "FLOWER-advanced-pytorch"


class AsyncBufferedFedAvgStrategy(FedAvg):
    """Asynchronous FedAvg with buffered replies before global updates."""

    def __init__(
        self,
        *args,
        train_timeout: Optional[float] = 60.0,
        reply_poll_interval: float = 1.0,
        async_max_in_flight: int = 4,
        async_evaluate_interval: int = 1,
        async_buffer_size: int = 4,
        lr_decay_interval: int = 20,
        lr_decay_factor: float = 0.9,
        min_learning_rate: float = 1e-4,
        staleness_weighting_enabled: bool = False,
        staleness_exponent: float = 1.0,
        target_accuracy: float = 0.20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_timeout = train_timeout
        self.reply_poll_interval = max(0.1, reply_poll_interval)
        self.async_max_in_flight = max(1, async_max_in_flight)
        self.async_evaluate_interval = max(1, async_evaluate_interval)
        self.async_buffer_size = max(1, async_buffer_size)
        self.lr_decay_interval = max(1, lr_decay_interval)
        self.lr_decay_factor = min(1.0, max(0.0, lr_decay_factor))
        self.min_learning_rate = max(0.0, min_learning_rate)
        self.staleness_weighting_enabled = staleness_weighting_enabled
        self.staleness_exponent = staleness_exponent
        
        # Tracking metrics for target accuracy
        self.target_accuracy = target_accuracy
        self.total_client_trips = 0
        self.start_time = time.time()
        self.target_reached = False

    def _maybe_decay_lr(self, global_updates_done: int, train_config: ConfigRecord) -> None:
        if global_updates_done <= 0 or global_updates_done % self.lr_decay_interval != 0:
            return
        if self.lr_decay_factor >= 1.0:
            return

        current_lr = float(train_config.get("lr", 0.0))
        if current_lr <= self.min_learning_rate:
            return

        new_lr = max(self.min_learning_rate, current_lr * self.lr_decay_factor)
        if new_lr >= current_lr:
            return

        train_config["lr"] = new_lr
        log(
            INFO,
            "[ASYNC-BUFFERED] LR decayed at global update %s: %.6f -> %.6f",
            global_updates_done,
            current_lr,
            new_lr,
        )

    def _check_target_accuracy(
        self,
        eval_metrics: Optional[MetricRecord],
        global_updates_done: int,
    ) -> None:
        """Check if target accuracy has been reached and log metrics if so.
        
        Args:
            eval_metrics: The evaluation metrics from global_evaluate_fn.
            global_updates_done: The current global update number.
        """
        if eval_metrics is None or self.target_reached:
            return
        
        # Extract accuracy from eval_metrics (supports 'accuracy' or similar keys)
        accuracy = None
        for key in ["accuracy", "acc", "test_accuracy", "eval_accuracy"]:
            if key in eval_metrics:
                accuracy = float(eval_metrics.get(key, 0.0))
                break
        
        if accuracy is None:
            return
        
        if accuracy >= self.target_accuracy:
            self.target_reached = True
            elapsed_time = time.time() - self.start_time
            
            # Log to wandb and console
            metrics_to_log = {
                "target_accuracy_reached": True,
                "wall_clock_time_to_target": elapsed_time,
                "client_trips_to_target": self.total_client_trips,
                "global_update_at_target": global_updates_done,
                "final_accuracy": accuracy,
            }
            
            log(
                INFO,
                "[ASYNC-BUFFERED] 🎯 TARGET ACCURACY REACHED! "
                "Accuracy: %.4f (target: %.4f) | "
                "Client Trips: %d | "
                "Wall-Clock Time: %.2f seconds | "
                "Global Updates: %d",
                accuracy,
                self.target_accuracy,
                self.total_client_trips,
                elapsed_time,
                global_updates_done,
            )
            
            wandb.log(metrics_to_log, step=global_updates_done)

    def _extract_train_reply(
        self,
        reply: Message,
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord], float]:
        if not reply.has_content():
            return None, None, 0.0
        if self.arrayrecord_key not in reply.content or "metrics" not in reply.content:
            return None, None, 0.0

        client_arrays = reply.content[self.arrayrecord_key]
        client_metrics = reply.content["metrics"]
        client_weight = float(client_metrics.get(self.weighted_by_key, 0.0))
        if client_weight <= 0:
            return None, None, 0.0

        return client_arrays, client_metrics, client_weight

    def _aggregate_buffered_replies(
        self,
        buffered_replies: list[tuple[Message, int]],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord], int]:
        if not buffered_replies:
            return None, None, 0

        # valid_updates: (client_arrays, client_metrics, effective_weight, tau, staleness_weight)
        valid_updates: list[tuple[ArrayRecord,
                                  MetricRecord, float, int, float]] = []
        for reply, tau in buffered_replies:
            client_arrays, client_metrics, num_examples = self._extract_train_reply(
                reply)
            if client_arrays is not None and client_metrics is not None and num_examples > 0:
                if self.staleness_weighting_enabled:
                    sw = polynomial_staleness_weight(
                        tau, self.staleness_exponent)
                    effective_weight = num_examples * sw
                else:
                    sw = 1.0
                    effective_weight = num_examples
                valid_updates.append(
                    (client_arrays, client_metrics, effective_weight, tau, sw))

        if not valid_updates:
            return None, None, 0

        total_weight = sum(w for _, _, w, _, _ in valid_updates)
        if total_weight <= 0:
            return None, None, 0

        aggregated_state: Optional[dict[str, torch.Tensor]] = None
        for client_arrays, _, client_weight, _, _ in valid_updates:
            client_state = client_arrays.to_torch_state_dict()
            if aggregated_state is None:
                aggregated_state = {
                    name: (
                        value * (client_weight / total_weight)
                        if torch.is_floating_point(value)
                        else value.clone()
                    )
                    for name, value in client_state.items()
                }
                continue

            for name, value in client_state.items():
                if torch.is_floating_point(value):
                    aggregated_state[name] = aggregated_state[name] + (
                        value * (client_weight / total_weight)
                    )
                else:
                    aggregated_state[name] = value.clone()

        if aggregated_state is None:
            return None, None, 0

        weighted_train_loss = 0.0
        for _, client_metrics, client_weight, _, _ in valid_updates:
            weighted_train_loss += float(
                client_metrics.get("train_loss", 0.0)) * client_weight

        metrics_dict: dict = {
            "train_loss": weighted_train_loss / total_weight,
            self.weighted_by_key: total_weight,
            "buffer_size": len(valid_updates),
        }
        if self.staleness_weighting_enabled:
            metrics_dict["avg_tau"] = sum(
                t for _, _, _, t, _ in valid_updates) / len(valid_updates)
            metrics_dict["avg_staleness_weight"] = sum(
                sw for _, _, _, _, sw in valid_updates) / len(valid_updates)

        return ArrayRecord(aggregated_state), MetricRecord(metrics_dict), len(valid_updates)

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
        exclude_node_ids: Optional[set[int]] = None,
        max_nodes_to_sample: Optional[int] = None,
    ) -> Iterable[Message]:
        from flwr.app import MessageType, RecordDict

        if self.fraction_train == 0.0:
            return []

        excluded = set() if exclude_node_ids is None else exclude_node_ids
        available_node_ids = [
            node_id for node_id in grid.get_node_ids() if node_id not in excluded]
        if not available_node_ids:
            return []

        fraction_target = max(
            1, int(len(available_node_ids) * self.fraction_train))
        min_train_nodes = max(1, int(getattr(self, "min_train_nodes", 1)))
        num_to_sample = max(fraction_target, min_train_nodes)
        num_to_sample = min(num_to_sample, len(available_node_ids))
        if max_nodes_to_sample is not None:
            if max_nodes_to_sample < min_train_nodes:
                log(
                    INFO,
                    "[ASYNC-BUFFERED] Sampling capped to %s (< min_train_nodes=%s)",
                    max_nodes_to_sample,
                    min_train_nodes,
                )
            num_to_sample = min(num_to_sample, max_nodes_to_sample)

        import random

        selected_node_ids = random.sample(available_node_ids, num_to_sample)

        config["server-round"] = server_round
        record = RecordDict({self.arrayrecord_key: arrays,
                            self.configrecord_key: config})
        return self._construct_messages(record, selected_node_ids, MessageType.TRAIN)

    def _dispatch_messages(
        self,
        *,
        grid: Grid,
        server_round: int,
        arrays: ArrayRecord,
        train_config: ConfigRecord,
        exclude_node_ids: Optional[set[int]],
        max_nodes_to_sample: int,
    ) -> dict[str, int]:
        messages = list(
            self.configure_train(
                server_round,
                arrays,
                train_config,
                grid,
                exclude_node_ids=exclude_node_ids,
                max_nodes_to_sample=max_nodes_to_sample,
            )
        )
        if not messages:
            return {}

        message_ids = list(grid.push_messages(messages))
        return {
            message_id: message.metadata.dst_node_id
            for message_id, message in zip(message_ids, messages)
        }

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[Callable[[int, ArrayRecord],
                                       Optional[MetricRecord]]] = None,
    ) -> Result:
        del timeout
        del evaluate_config

        wandb.init(project=PROJECT_NAME)

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, ConfigRecord())
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        result = Result()
        result.arrays = initial_arrays

        try:
            t_start = time.time()
            arrays = initial_arrays
            next_server_round = 1

            if evaluate_fn is not None:
                initial_res = evaluate_fn(0, arrays)
                log(INFO, "Initial global evaluation results: %s", initial_res)
                if initial_res is not None:
                    result.evaluate_metrics_serverapp[0] = initial_res
                    # Check if target accuracy is already reached
                    self._check_target_accuracy(initial_res, 0)

            pending_message_ids: set[str] = set()
            pending_node_ids: set[int] = set()
            message_to_node: dict[str, int] = {}
            message_to_dispatch_update: dict[str, int] = {}
            buffered_replies: list[tuple[Message, int]] = []
            global_updates_done = 0

            initial_dispatch = self._dispatch_messages(
                grid=grid,
                server_round=next_server_round,
                arrays=arrays,
                train_config=train_config,
                exclude_node_ids=None,
                max_nodes_to_sample=self.async_max_in_flight,
            )
            pending_message_ids.update(initial_dispatch)
            for message_id, node_id in initial_dispatch.items():
                pending_node_ids.add(node_id)
                message_to_node[message_id] = node_id
                message_to_dispatch_update[message_id] = global_updates_done
            next_server_round += len(initial_dispatch)

            if not pending_message_ids:
                log(INFO, "[ASYNC-BUFFERED] No train messages dispatched. Exiting.")
                return result

            last_progress = time.monotonic()
            while global_updates_done < max(1, num_rounds):
                replies = list(grid.pull_messages(list(pending_message_ids)))
                if not replies:
                    if (
                        self.train_timeout is not None
                        and (time.monotonic() - last_progress) >= self.train_timeout
                    ):
                        log(
                            INFO,
                            "[ASYNC-BUFFERED] Idle timeout reached with %s pending replies. Waiting for late clients.",
                            len(pending_message_ids),
                        )
                        last_progress = time.monotonic()
                        continue
                    time.sleep(self.reply_poll_interval)
                    continue

                last_progress = time.monotonic()
                for reply in replies:
                    reply_to_message_id = reply.metadata.reply_to_message_id
                    pending_message_ids.discard(reply_to_message_id)

                    node_id = message_to_node.pop(reply_to_message_id, None)
                    if node_id is not None:
                        pending_node_ids.discard(node_id)

                    client_arrays, client_metrics, client_weight = self._extract_train_reply(
                        reply)
                    if client_arrays is None or client_metrics is None or client_weight <= 0:
                        continue

                    dispatch_update = message_to_dispatch_update.pop(
                        reply_to_message_id, 0)
                    tau = global_updates_done - dispatch_update
                    buffered_replies.append((reply, tau))
                    
                    # Increment client trips counter
                    self.total_client_trips += 1
                    
                    log(
                        INFO,
                        "[ASYNC-BUFFERED] buffered reply from node %s tau=%d (%s/%s) [Client Trips: %d]",
                        reply.metadata.src_node_id,
                        tau,
                        len(buffered_replies),
                        self.async_buffer_size,
                        self.total_client_trips,
                    )

                    if len(buffered_replies) >= self.async_buffer_size:
                        agg_arrays, agg_metrics, consumed = self._aggregate_buffered_replies(
                            buffered_replies)
                        buffered_replies = []

                        if agg_arrays is not None and agg_metrics is not None and consumed > 0:
                            global_updates_done += 1
                            arrays = agg_arrays
                            result.arrays = agg_arrays
                            result.train_metrics_clientapp[global_updates_done] = agg_metrics
                            self._maybe_decay_lr(
                                global_updates_done, train_config)
                            wandb.log(dict(agg_metrics),
                                      step=global_updates_done)
                            log(
                                INFO,
                                "[ASYNC-BUFFERED AGG %s/%s] applied buffered FedAvg with %s replies",
                                global_updates_done,
                                num_rounds,
                                consumed,
                            )

                            if evaluate_fn is not None and global_updates_done % self.async_evaluate_interval == 0:
                                eval_res = evaluate_fn(
                                    global_updates_done, arrays)
                                log(INFO, "\t└──> MetricRecord: %s", eval_res)
                                if eval_res is not None:
                                    result.evaluate_metrics_serverapp[global_updates_done] = eval_res
                                    wandb.log(dict(eval_res),
                                              step=global_updates_done)
                                    # Check if target accuracy has been reached
                                    self._check_target_accuracy(eval_res, global_updates_done)

                            if global_updates_done >= max(1, num_rounds):
                                break

                    refill = self._dispatch_messages(
                        grid=grid,
                        server_round=next_server_round,
                        arrays=arrays,
                        train_config=train_config,
                        exclude_node_ids=pending_node_ids,
                        max_nodes_to_sample=1,
                    )
                    pending_message_ids.update(refill)
                    for mid, nid in refill.items():
                        pending_node_ids.add(nid)
                        message_to_node[mid] = nid
                        message_to_dispatch_update[mid] = global_updates_done
                    next_server_round += len(refill)

                if global_updates_done >= max(1, num_rounds):
                    break

            if buffered_replies and global_updates_done < max(1, num_rounds):
                agg_arrays, agg_metrics, consumed = self._aggregate_buffered_replies(
                    buffered_replies)
                if agg_arrays is not None and agg_metrics is not None and consumed > 0:
                    global_updates_done += 1
                    arrays = agg_arrays
                    result.arrays = agg_arrays
                    result.train_metrics_clientapp[global_updates_done] = agg_metrics
                    self._maybe_decay_lr(global_updates_done, train_config)
                    wandb.log(dict(agg_metrics), step=global_updates_done)

                    if evaluate_fn is not None and global_updates_done % self.async_evaluate_interval == 0:
                        eval_res = evaluate_fn(global_updates_done, arrays)
                        log(INFO, "\t└──> MetricRecord: %s", eval_res)
                        if eval_res is not None:
                            result.evaluate_metrics_serverapp[global_updates_done] = eval_res
                            wandb.log(dict(eval_res), step=global_updates_done)
                            # Check if target accuracy has been reached
                            self._check_target_accuracy(eval_res, global_updates_done)

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
