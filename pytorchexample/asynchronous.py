import io
import time
from contextlib import suppress
from logging import INFO
from pprint import pformat
from typing import Any, Callable, Iterable, Optional

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

from pytorchexample.staleness import polynomial_staleness_weight
from pytorchexample.task import (
    compute_direction_similarity,
    compute_lora_update_norm,
    extract_lora_state,
    get_top1_test_accuracy,
)

PROJECT_NAME = "FL LoRA"


class AsyncFedAvgStrategy(FedAvg):
    """Asynchronous FedAvg with immediate global updates per client reply."""

    def __init__(
        self,
        *args,
        train_timeout: Optional[float] = 60.0,
        reply_poll_interval: float = 1.0,
        async_max_in_flight: int = 4,
        async_evaluate_interval: int = 1,
        lr_decay_interval: int = 20,
        lr_decay_factor: float = 0.9,
        min_learning_rate: float = 1e-4,
        staleness_weighting_mode: str = "polynomial",
        fedstaleweight_ema_beta: float = 0.8,
        staleness_weighting_enabled: bool = False,
        staleness_exponent: float = 1.0,
        run_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_timeout = train_timeout
        self.reply_poll_interval = max(0.1, reply_poll_interval)
        self.async_max_in_flight = max(1, async_max_in_flight)
        self.async_evaluate_interval = max(1, async_evaluate_interval)
        self.lr_decay_interval = max(1, lr_decay_interval)
        self.lr_decay_factor = min(1.0, max(0.0, lr_decay_factor))
        self.min_learning_rate = max(0.0, min_learning_rate)
        self.staleness_weighting_mode = str(
            staleness_weighting_mode).strip().lower()
        self.fedstaleweight_ema_beta = min(
            0.99, max(0.0, fedstaleweight_ema_beta))
        self.staleness_weighting_enabled = staleness_weighting_enabled
        self.staleness_exponent = staleness_exponent
        self.client_staleness_ema: dict[int, float] = {}
        self.run_config = dict(run_config) if run_config is not None else {}

    def _log_run_config(
        self,
        stage: str,
        step: int,
        train_config: Optional[ConfigRecord],
    ) -> None:
        if stage == "start":
            config_payload: dict[str, Any] = {}
            if self.run_config:
                config_payload.update(
                    {f"run_config.{k}": v for k, v in self.run_config.items()}
                )
            if train_config is not None:
                config_payload.update(
                    {f"train_config.{k}": v for k,
                        v in dict(train_config).items()}
                )
            if config_payload:
                wandb.config.update(config_payload, allow_val_change=True)

        log(INFO, "[CONFIG-%s] run_config=%s",
            stage.upper(), pformat(self.run_config))
        if train_config is not None:
            log(
                INFO,
                "[CONFIG-%s] train_config=%s",
                stage.upper(),
                pformat(dict(train_config)),
            )

    def _maybe_decay_lr(self, updates_done: int, train_config: ConfigRecord) -> None:
        if updates_done <= 0 or updates_done % self.lr_decay_interval != 0:
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
            "[ASYNC] LR decayed at update %s: %.6f -> %.6f",
            updates_done,
            current_lr,
            new_lr,
        )

    def _update_staleness_ema(self, node_id: int, tau: int) -> float:
        previous = self.client_staleness_ema.get(node_id, float(tau))
        updated = (
            self.fedstaleweight_ema_beta * previous
            + (1.0 - self.fedstaleweight_ema_beta) * float(tau)
        )
        self.client_staleness_ema[node_id] = updated
        return updated

    def _compute_effective_weight(
        self,
        *,
        node_id: int,
        tau: int,
        num_examples: float,
    ) -> tuple[float, float]:
        if not self.staleness_weighting_enabled:
            return num_examples, 1.0

        mode = self.staleness_weighting_mode
        if mode in {"fedstaleweight", "fair"}:
            expected_staleness = self._update_staleness_ema(node_id, tau)
            fairness_boost = 1.0 + \
                (max(0.0, expected_staleness) / max(1, self.async_max_in_flight))
            return num_examples * fairness_boost, fairness_boost

        if mode in {"polynomial", "poly"}:
            fairness_boost = polynomial_staleness_weight(
                tau, self.staleness_exponent)
            return num_examples * fairness_boost, fairness_boost

        raise ValueError(
            "Invalid staleness-weighting-mode. Use 'polynomial' or 'fedstaleweight'."
        )

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
        node_id: int,
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

        effective_weight, staleness_boost = self._compute_effective_weight(
            node_id=node_id,
            tau=tau,
            num_examples=num_examples,
        )

        current_state = current_arrays.to_torch_state_dict()
        client_state = client_arrays.to_torch_state_dict()
        update_norm = compute_lora_update_norm(
            extract_lora_state(client_state),
            extract_lora_state(current_state),
        )

        if current_total_weight <= 0:
            if self.staleness_weighting_enabled:
                client_metrics = MetricRecord(
                    {**client_metrics, "staleness_weight": staleness_boost}
                )
            if update_norm is not None:
                client_metrics = MetricRecord(
                    {**client_metrics, "client_update_norm": update_norm}
                )
            return client_arrays, effective_weight, client_metrics

        merged_state = self._weighted_average_state_dicts(
            current_state,
            current_total_weight,
            client_state,
            effective_weight,
        )
        if self.staleness_weighting_enabled:
            client_metrics = MetricRecord(
                {**client_metrics, "staleness_weight": staleness_boost}
            )
        if update_norm is not None:
            client_metrics = MetricRecord(
                {**client_metrics, "client_update_norm": update_norm}
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
                    "[ASYNC] Sampling capped to %s (< min_train_nodes=%s)",
                    max_nodes_to_sample,
                    min_train_nodes,
                )
            num_to_sample = min(num_to_sample, max_nodes_to_sample)

        import random

        selection_seed = int(self.run_config.get("selection-seed", 42))
        rng = random.Random(selection_seed + int(server_round))
        selected_node_ids = rng.sample(sorted(available_node_ids), num_to_sample)

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
        stop_mode: str = "num_rounds",
        target_accuracy: float = 0.9,
        run_name: Optional[str] = None,
    ) -> Result:
        del timeout
        del evaluate_config

        wandb.init(project=PROJECT_NAME, name=run_name or None)

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, ConfigRecord())
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        result = Result()
        result.arrays = initial_arrays
        target_mode = str(stop_mode).strip().lower() in {
            "target_accuracy",
            "target-accuracy",
            "target",
        }
        evaluation_interval = 1 if target_mode else self.async_evaluate_interval

        self._log_run_config("start", 0, train_config)

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
            last_step = 0
            t_start = time.time()
            last_update_time = time.time()
            arrays = initial_arrays
            weighted_examples_seen = 0.0
            client_trips_done = 0
            target_logged = False
            peak_accuracy = 0.0
            previous_lora_state = extract_lora_state(
                initial_arrays.to_torch_state_dict()
            )

            if evaluate_fn is not None:
                initial_res = evaluate_fn(0, arrays)
                log(INFO, "Initial global evaluation results: %s", initial_res)
                if initial_res is not None:
                    result.evaluate_metrics_serverapp[0] = initial_res
                    initial_log = dict(initial_res)
                    if "eval_loss" in initial_log:
                        initial_log["global_eval_loss"] = initial_log.pop(
                            "eval_loss")
                    if "top1_test_accuracy" in initial_log:
                        initial_log["baseline_top1_test_accuracy"] = initial_log.pop(
                            "top1_test_accuracy"
                        )
                    wandb.log(initial_log, step=0)
                    accuracy = get_top1_test_accuracy(initial_res)
                    if accuracy is not None and accuracy >= target_accuracy and not target_logged:
                        log_target_metrics(0, client_trips_done)
                        target_logged = True
                    if target_mode and accuracy is not None and accuracy >= target_accuracy:
                        return result

            target_updates = max(1, num_rounds)
            updates_done = 0
            next_server_round = 1

            tau_history: list[int] = []
            tau_by_tier: dict[str, list[int]] = {
                "FAST": [], "MEDIUM": [], "SLOW": []}

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

            _dispatch_miss_total = 0
            last_progress = time.monotonic()
            while updates_done < target_updates:
                replies = list(grid.pull_messages(list(pending_message_ids)))
                if not replies:
                    if (
                        self.train_timeout is not None
                        and (time.monotonic() - last_progress) >= self.train_timeout
                    ):
                        log(
                            INFO,
                            "[ASYNC] Idle timeout reached with %s pending replies. Waiting for late clients.",
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

                    _reply_id_found = reply_to_message_id in message_to_dispatch_update
                    if not _reply_id_found:
                        _dispatch_miss_total += 1
                    dispatch_update = message_to_dispatch_update.pop(
                        reply_to_message_id, 0)
                    tau = updates_done - dispatch_update

                    agg_arrays, weighted_examples_seen, train_metrics = self._apply_async_fedavg_update(
                        arrays,
                        weighted_examples_seen,
                        reply,
                        node_id=node_id if node_id is not None else reply.metadata.src_node_id,
                        tau=tau,
                    )
                    if agg_arrays is None or train_metrics is None:
                        continue

                    if updates_done < 30:
                        _raw = reply.content.get(
                            "metrics") if reply.has_content() else None
                        _stier = float(
                            _raw.get("straggler_tier_id", -1)) if _raw else -1.0
                        _ssleep = float(
                            _raw.get("straggler_sleep_seconds", -1.0)) if _raw else -1.0
                        log(
                            INFO,
                            "[TAU-DEBUG #%d] dispatch_update=%d updates_done_before=%d tau=%d "
                            "node=%s reply_id_found=%s in_flight=%d "
                            "straggler_tier_id=%.0f straggler_sleep=%.2fs",
                            updates_done,
                            dispatch_update,
                            updates_done,
                            tau,
                            node_id if node_id is not None else reply.metadata.src_node_id,
                            _reply_id_found,
                            len(pending_message_ids),
                            _stier,
                            _ssleep,
                        )

                    updates_done += 1
                    last_step = updates_done
                    client_trips_done += 1
                    arrays = agg_arrays
                    result.arrays = agg_arrays
                    result.train_metrics_clientapp[updates_done] = train_metrics
                    self._maybe_decay_lr(updates_done, train_config)

                    current_state = arrays.to_torch_state_dict()
                    direction_similarity = compute_direction_similarity(
                        current_state,
                        previous_lora_state,
                    )
                    previous_lora_state = extract_lora_state(current_state)

                    now = time.time()
                    round_duration = now - last_update_time
                    last_update_time = now

                    tau_history.append(tau)
                    _id_to_tier = {0: "FAST", 1: "MEDIUM", 2: "SLOW"}
                    tier_label = _id_to_tier.get(
                        int(train_metrics.get("straggler_tier_id", -1)), "")
                    if tier_label in tau_by_tier:
                        tau_by_tier[tier_label].append(tau)

                    log_dict = dict(train_metrics)
                    log_dict["round_duration"] = round_duration
                    log_dict["tau"] = tau
                    if direction_similarity is not None:
                        log_dict["direction_similarity"] = direction_similarity
                    if self.staleness_weighting_enabled:
                        num_examples = float(
                            train_metrics.get(self.weighted_by_key, 0.0))
                        staleness_boost = float(
                            train_metrics.get("staleness_weight", 1.0))
                        final_weight = num_examples * staleness_boost
                        log(
                            INFO,
                            "[STALENESS] mode=%s tau=%d staleness_boost=%.4f final_weight=%.2f",
                            self.staleness_weighting_mode, tau, staleness_boost, final_weight,
                        )
                        log_dict.update({
                            "staleness_weight": staleness_boost,
                            "final_weight": final_weight,
                        })
                    wandb.log(log_dict, step=updates_done)
                    log(INFO, "[ASYNC UPDATE %s/%s] integrated reply from node %s",
                        updates_done, target_updates, reply.metadata.src_node_id)

                    if tau_history and updates_done % evaluation_interval == 0:
                        import statistics
                        staleness_log = {
                            "staleness/mean_tau": statistics.mean(tau_history),
                            "staleness/median_tau": statistics.median(tau_history),
                            "staleness/max_tau": max(tau_history),
                            "staleness/tau_dist": wandb.Histogram(tau_history),
                        }
                        for tier_name, tier_taus in tau_by_tier.items():
                            if tier_taus:
                                staleness_log[f"staleness/mean_tau_{tier_name.lower()}"] = statistics.mean(tier_taus)
                        wandb.log(staleness_log, step=updates_done)

                    if evaluate_fn is not None and updates_done % evaluation_interval == 0:
                        eval_res = evaluate_fn(updates_done, arrays)
                        log(INFO, "\t└──> MetricRecord: %s", eval_res)
                        if eval_res is not None:
                            result.evaluate_metrics_serverapp[updates_done] = eval_res
                            global_eval_log = dict(eval_res)
                            if "eval_loss" in global_eval_log:
                                global_eval_log["global_eval_loss"] = global_eval_log.pop(
                                    "eval_loss")
                            wandb.log(global_eval_log, step=updates_done)
                            accuracy = get_top1_test_accuracy(eval_res)
                            if accuracy is not None:
                                if accuracy > peak_accuracy:
                                    peak_accuracy = accuracy
                                    wandb.run.summary["peak_top1_test_accuracy"] = accuracy
                                    wandb.run.summary["peak_accuracy_step"] = updates_done
                                    wandb.run.summary["peak_accuracy_wall_clock_seconds"] = time.time(
                                    ) - t_start
                                wandb.log(
                                    {"peak_top1_test_accuracy": peak_accuracy}, step=updates_done)
                            if accuracy is not None and accuracy >= target_accuracy:
                                if not target_logged:
                                    target_logged = True
                                    log_target_metrics(
                                        updates_done, client_trips_done)
                                if target_mode:
                                    return result

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

            if _dispatch_miss_total > 0:
                log(
                    INFO,
                    "[TAU-DEBUG] %d/%d reply IDs not found in dispatch map (used default dispatch_update=0)",
                    _dispatch_miss_total,
                    updates_done,
                )
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
                self._log_run_config("end", last_step, train_config)
                wandb.finish()
