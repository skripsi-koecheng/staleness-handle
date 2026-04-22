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


class AsyncFedAvgStrategy(FedAvg):
    """Asynchronous FedAvg with immediate global updates per client reply."""

    def __init__(
        self,
        *args,
        train_timeout: Optional[float] = 60.0,
        reply_poll_interval: float = 1.0,
        async_max_in_flight: int = 4,
        async_evaluate_interval: int = 1,
        staleness_weighting_enabled: bool = False,
        staleness_exponent: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_timeout = train_timeout
        self.reply_poll_interval = max(0.1, reply_poll_interval)
        self.async_max_in_flight = max(1, async_max_in_flight)
        self.async_evaluate_interval = max(1, async_evaluate_interval)
        self.staleness_weighting_enabled = staleness_weighting_enabled
        self.staleness_exponent = staleness_exponent

    @staticmethod
    def _weighted_average_state_dicts(
        base_state: dict[str, torch.Tensor],
        base_weight: float,
        client_state: dict[str, torch.Tensor],
        client_weight: float,
    ) -> dict[str, torch.Tensor]:
        total_weight = base_weight + client_weight
        if total_weight <= 0:
            return base_state

        merged_state: dict[str, torch.Tensor] = {}
        for name, base_value in base_state.items():
            client_value = client_state[name]
            if torch.is_floating_point(base_value):
                merged_state[name] = (
                    base_value * (base_weight / total_weight)
                    + client_value * (client_weight / total_weight)
                )
            else:
                merged_state[name] = client_value
        return merged_state

    def _apply_async_fedavg_update(
        self,
        current_arrays: ArrayRecord,
        current_total_weight: float,
        reply: Message,
        tau: int = 0,
    ) -> tuple[Optional[ArrayRecord], float, Optional[MetricRecord]]:
        if not reply.has_content():
            return None, current_total_weight, None
        if self.arrayrecord_key not in reply.content or "metrics" not in reply.content:
            return None, current_total_weight, None

        client_arrays = reply.content[self.arrayrecord_key]
        client_metrics = reply.content["metrics"]
        num_examples = float(client_metrics.get(self.weighted_by_key, 0.0))
        if num_examples <= 0:
            return None, current_total_weight, None

        if self.staleness_weighting_enabled:
            effective_weight = num_examples * polynomial_staleness_weight(tau, self.staleness_exponent)
        else:
            effective_weight = num_examples

        if current_total_weight <= 0:
            return client_arrays, effective_weight, client_metrics

        current_state = current_arrays.to_torch_state_dict()
        client_state = client_arrays.to_torch_state_dict()
        merged_state = self._weighted_average_state_dicts(
            current_state,
            current_total_weight,
            client_state,
            effective_weight,
        )
        return ArrayRecord(merged_state), current_total_weight + effective_weight, client_metrics

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

        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.5
            log(INFO, "LR decreased to: %s", config["lr"])

        excluded = set() if exclude_node_ids is None else exclude_node_ids
        available_node_ids = [node_id for node_id in grid.get_node_ids() if node_id not in excluded]
        if not available_node_ids:
            return []

        num_to_sample = max(1, int(len(available_node_ids) * self.fraction_train))
        num_to_sample = min(num_to_sample, len(available_node_ids))
        if max_nodes_to_sample is not None:
            num_to_sample = min(num_to_sample, max_nodes_to_sample)

        import random

        selected_node_ids = random.sample(available_node_ids, num_to_sample)

        config["server-round"] = server_round
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
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
        evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None,
    ) -> Result:
        del timeout
        del evaluate_config

        wandb.init(project=PROJECT_NAME)

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(num_rounds, initial_arrays, train_config, ConfigRecord())
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        result = Result()
        result.arrays = initial_arrays

        try:
            t_start = time.time()
            arrays = initial_arrays
            weighted_examples_seen = 0.0

            if evaluate_fn is not None:
                initial_res = evaluate_fn(0, arrays)
                log(INFO, "Initial global evaluation results: %s", initial_res)
                if initial_res is not None:
                    result.evaluate_metrics_serverapp[0] = initial_res

            target_updates = max(1, num_rounds)
            updates_done = 0
            next_server_round = 1

            pending_message_ids: set[str] = set()
            pending_node_ids: set[int] = set()
            message_to_node: dict[str, int] = {}
            message_to_dispatch_update: dict[str, int] = {}

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
                message_to_dispatch_update[message_id] = updates_done
            next_server_round += len(initial_dispatch)

            if not pending_message_ids:
                log(INFO, "[ASYNC] No train messages dispatched. Exiting.")
                return result

            last_progress = time.monotonic()
            while updates_done < target_updates:
                replies = list(grid.pull_messages(list(pending_message_ids)))
                if not replies:
                    if (
                        self.train_timeout is not None
                        and (time.monotonic() - last_progress) >= self.train_timeout
                    ):
                        log(INFO, "[ASYNC] Idle timeout reached.")
                        break
                    time.sleep(self.reply_poll_interval)
                    continue

                last_progress = time.monotonic()
                for reply in replies:
                    reply_to_message_id = reply.metadata.reply_to_message_id
                    pending_message_ids.discard(reply_to_message_id)

                    node_id = message_to_node.pop(reply_to_message_id, None)
                    if node_id is not None:
                        pending_node_ids.discard(node_id)

                    dispatch_update = message_to_dispatch_update.pop(reply_to_message_id, 0)
                    tau = updates_done - dispatch_update

                    agg_arrays, weighted_examples_seen, train_metrics = self._apply_async_fedavg_update(
                        arrays,
                        weighted_examples_seen,
                        reply,
                        tau=tau,
                    )
                    if agg_arrays is None or train_metrics is None:
                        continue

                    updates_done += 1
                    arrays = agg_arrays
                    result.arrays = agg_arrays
                    result.train_metrics_clientapp[updates_done] = train_metrics

                    log_dict = dict(train_metrics)
                    if self.staleness_weighting_enabled:
                        staleness_weight = polynomial_staleness_weight(tau, self.staleness_exponent)
                        num_examples = float(train_metrics.get(self.weighted_by_key, 0.0))
                        final_weight = num_examples * staleness_weight
                        log(
                            INFO,
                            "[STALENESS] tau=%d staleness_weight=%.4f final_weight=%.2f",
                            tau, staleness_weight, final_weight,
                        )
                        log_dict.update({
                            "tau": tau,
                            "staleness_weight": staleness_weight,
                            "final_weight": final_weight,
                        })
                    wandb.log(log_dict, step=updates_done)
                    log(INFO, "[ASYNC UPDATE %s/%s] integrated reply from node %s", updates_done, target_updates, reply.metadata.src_node_id)

                    if evaluate_fn is not None and updates_done % self.async_evaluate_interval == 0:
                        eval_res = evaluate_fn(updates_done, arrays)
                        log(INFO, "\t└──> MetricRecord: %s", eval_res)
                        if eval_res is not None:
                            result.evaluate_metrics_serverapp[updates_done] = eval_res
                            wandb.log(dict(eval_res), step=updates_done)

                    if updates_done >= target_updates:
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
                        message_to_dispatch_update[mid] = updates_done
                    next_server_round += len(refill)

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
