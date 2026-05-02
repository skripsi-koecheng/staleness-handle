import torch
import torch.nn as nn
from datasets import load_dataset
from flwr.app import ArrayRecord, MetricRecord
from flwr_datasets import FederatedDataset
import torch.nn.functional as F
from flwr_datasets.partitioner import IidPartitioner
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.utils.data import DataLoader
import os
import random
import numpy as np
from logging import INFO
from typing import Optional

from flwr.common import log
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

DATASET_NAME = "sh0416/ag_news"
MODEL_NAME = "distilbert/distilbert-base-uncased"
NUM_LABELS = 4
MAX_LENGTH = 256
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_lin", "v_lin"]

fds = None
tokenizer = None

TEXT_CANDIDATE_KEYS = ["text", "sentence", "content", "article", "description"]
LABEL_CANDIDATE_KEYS = ["label", "labels", "class", "category", "topic"]
TOP1_TEST_ACCURACY_KEY = "top1_test_accuracy"
DIR_VARIATION_KEY = "direction_variation"

# Global seed for deterministic runs
GLOBAL_MODEL_SEED = 42


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = GLOBAL_MODEL_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class DistilBertAgNewsClassifier(nn.Module):
    """DistilBERT + LoRA adapter for AG News text classification."""

    def __init__(self):
        super().__init__()
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        self.model = get_peft_model(base_model, peft_config)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def get_federated_state_dict(self):
        return get_peft_model_state_dict(self.model)

    def load_federated_state_dict(self, state_dict):
        set_peft_model_state_dict(self.model, state_dict)


def _get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def _collate_batch(batch):
    tok = _get_tokenizer()

    sample = batch[0]
    text_key = next(
        (key for key in TEXT_CANDIDATE_KEYS if key in sample), None)
    label_key = next(
        (key for key in LABEL_CANDIDATE_KEYS if key in sample), None)

    if "title" in sample and "description" in sample:
        texts = [f"{item['title']} {item['description']}" for item in batch]
    elif text_key is not None:
        texts = [str(item[text_key]) for item in batch]
    else:
        raise KeyError(
            "No text column found in batch. "
            f"Available keys: {list(sample.keys())}"
        )

    if label_key is None:
        raise KeyError(
            "No label column found in batch. "
            f"Available keys: {list(sample.keys())}"
        )

    labels = torch.tensor([int(item[label_key])
                          for item in batch], dtype=torch.long)
    if labels.min().item() >= 1 and labels.max().item() == NUM_LABELS:
        labels = labels - 1

    if labels.min().item() < 0 or labels.max().item() >= NUM_LABELS:
        raise ValueError(
            f"Label values out of range after normalization: "
            f"min={labels.min().item()}, max={labels.max().item()}, "
            f"expected in [0, {NUM_LABELS - 1}]"
        )

    encoded = tok(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded["labels"] = labels
    return encoded


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load IID partition of AG News and return local train/val loaders."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset=DATASET_NAME,
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(
        test_size=0.2, seed=GLOBAL_MODEL_SEED + partition_id
    )
    gen = torch.Generator()
    gen.manual_seed(GLOBAL_MODEL_SEED + partition_id)

    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
        generator=gen,
        worker_init_fn=seed_worker,
    )
    testloader = DataLoader(
        partition_train_test["test"],
        batch_size=batch_size,
        collate_fn=_collate_batch,
        generator=gen,
        worker_init_fn=seed_worker,
    )
    return trainloader, testloader


def load_centralized_dataset():
    """Load central AG News test split and return dataloader."""
    test_dataset = load_dataset(DATASET_NAME, split="test")
    return DataLoader(test_dataset, batch_size=128, collate_fn=_collate_batch)


def train(
    net,
    trainloader,
    epochs,
    lr,
    device,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
):
    """Train DistilBERT locally."""
    net.to(device)
    trainable_parameters = [
        param for param in net.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=lr,
        weight_decay=weight_decay,
    )
    total_steps = max(1, epochs * len(trainloader))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = net(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Evaluate DistilBERT."""
    net.to(device)
    net.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = net(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_top1_test_accuracy(metrics: MetricRecord) -> Optional[float]:
    value = metrics.get(TOP1_TEST_ACCURACY_KEY)
    if value is None:
        return None
    return float(value)


def _is_lora_tensor(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and tensor.is_floating_point()
        and ("lora_A" in name or "lora_B" in name)
    )


def _average_column_cosine_similarity(
    current_state: dict[str, torch.Tensor],
    previous_state: dict[str, torch.Tensor],
) -> Optional[float]:
    similarities: list[float] = []
    for name, current_tensor in current_state.items():
        previous_tensor = previous_state.get(name)
        if previous_tensor is None or current_tensor.shape != previous_tensor.shape:
            continue
        if not _is_lora_tensor(name, current_tensor):
            continue

        for column_idx in range(current_tensor.shape[1]):
            current_column = current_tensor[:, column_idx].float().reshape(-1)
            previous_column = previous_tensor[:,
                                              column_idx].float().reshape(-1)
            similarity = F.cosine_similarity(
                current_column,
                previous_column,
                dim=0,
                eps=1e-8,
            )
            similarities.append(float(similarity.item()))

    if not similarities:
        return None
    return sum(similarities) / len(similarities)


def extract_lora_state(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
        if _is_lora_tensor(name, tensor)
    }


def compute_direction_variation(
    current_state: dict[str, torch.Tensor],
    previous_state: Optional[dict[str, torch.Tensor]],
) -> Optional[float]:
    if previous_state is None:
        return None
    return _average_column_cosine_similarity(current_state, previous_state)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on centralized AG News test data."""
    set_global_seed(GLOBAL_MODEL_SEED)
    model = DistilBertAgNewsClassifier()
    model.load_federated_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataloader = load_centralized_dataset()
    _, test_acc = test(model, test_dataloader, device)
    log(
        INFO,
        "[GLOBAL][ROUND %s] Top-1 Test Accuracy=%.4f",
        server_round,
        test_acc,
    )
    return MetricRecord({TOP1_TEST_ACCURACY_KEY: test_acc})
