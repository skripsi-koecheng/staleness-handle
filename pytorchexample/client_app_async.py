"""pytorchexample async client app."""

import time
from logging import INFO

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import log

from pytorchexample.straggler import compute_straggler, describe_tier_mapping
from pytorchexample.task import (
    DistilBertAgNewsClassifier,
    get_state_dict_bytes,
    get_state_dict_numel,
    load_data,
    set_global_seed,
    GLOBAL_MODEL_SEED,
)
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    partition_id = context.node_config["partition-id"]
    # Seed deterministically per client partition
    set_global_seed(GLOBAL_MODEL_SEED + partition_id)
    use_lora = bool(context.run_config.get("use-lora", True))
    model = DistilBertAgNewsClassifier(use_lora=use_lora)
    model.load_federated_state_dict(
        msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    dirichlet_alpha = float(context.run_config.get("dirichlet-alpha", 0.25))
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, dirichlet_alpha)

    train_start = time.perf_counter()
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        msg.content["config"].get("weight_decay", 0.01),
        msg.content["config"].get("warmup_ratio", 0.1),
        msg.content["config"].get("max_grad_norm", 1.0),
    )
    train_duration = time.perf_counter() - train_start

    straggler_metrics: dict = {}
    straggler_enabled = bool(
        context.run_config.get("straggler-enabled", False))
    if straggler_enabled:
        baseline_mode = str(context.run_config.get(
            "baseline-mode", "measured"))
        scenario = str(context.run_config.get(
            "straggler-scenario", "balanced"))

        if baseline_mode == "measured":
            baseline_t = train_duration
        else:
            baseline_t = float(context.run_config.get(
                "baseline-time-seconds", 60.0))

        if partition_id == 0:
            log(INFO, "[STRAGGLER] Tier map:\n%s",
                describe_tier_mapping(num_partitions, scenario))

        sim = compute_straggler(
            partition_id=partition_id,
            total_clients=num_partitions,
            scenario=scenario,
            baseline_mode=baseline_mode,
            baseline_T=baseline_t,
            train_duration=train_duration,
        )

        log(
            INFO,
            "[STRAGGLER] partition=%d | scenario=%s | tier=%s | mult=%.1fx | mode=%s | T=%.2fs | train=%.2fs | target=%.2fs | sleep=%.2fs | total=%.2fs",
            sim.partition_id,
            sim.scenario,
            sim.tier,
            sim.multiplier,
            sim.baseline_mode,
            sim.baseline_T,
            sim.train_duration,
            sim.target_completion_time,
            sim.sleep_duration,
            sim.total_effective_time,
        )

        _tier_to_id = {"FAST": 0, "MEDIUM": 1, "SLOW": 2}
        straggler_metrics = {
            "straggler_tier_id": _tier_to_id.get(sim.tier, -1),
            "straggler_multiplier": sim.multiplier,
            "straggler_sleep_seconds": sim.sleep_duration,
            "straggler_total_effective_time": sim.total_effective_time,
        }

        if sim.sleep_duration > 0:
            time.sleep(sim.sleep_duration)

    state_dict = model.get_federated_state_dict()
    comm_params = get_state_dict_numel(state_dict)
    comm_bytes = get_state_dict_bytes(state_dict)
    full_bytes = model.get_full_state_bytes()
    relative_ratio = comm_bytes / full_bytes if full_bytes > 0 else 0.0
    model_record = ArrayRecord(state_dict)
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "dispatched-server-round": int(msg.content["config"].get("server-round", 0)),
        **straggler_metrics,
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        metrics.update(
            {
                "vram_allocated_mb": torch.cuda.memory_allocated(device)
                / (1024**2),
                "vram_reserved_mb": torch.cuda.memory_reserved(device)
                / (1024**2),
            }
        )
    metrics.update(
        {
            "communication_bytes": comm_bytes,
            "communication_megabytes": comm_bytes / (1024**2),
            "communication_params": comm_params,
            "relative_bandwidth_ratio": relative_ratio,
        }
    )
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    partition_id = context.node_config["partition-id"]
    # Seed deterministically per client partition
    set_global_seed(GLOBAL_MODEL_SEED + partition_id)
    use_lora = bool(context.run_config.get("use-lora", True))
    model = DistilBertAgNewsClassifier(use_lora=use_lora)
    model.load_federated_state_dict(
        msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    dirichlet_alpha = float(context.run_config.get("dirichlet-alpha", 0.25))
    _, valloader = load_data(partition_id, num_partitions, batch_size, dirichlet_alpha)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
