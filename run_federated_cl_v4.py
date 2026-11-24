#!/usr/bin/env python3
"""V4 federated continual-learning runner with isolated, auditable components."""

import copy
import csv
import json
import logging
import os
import pickle
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from config import CFG, CURRENT_DATASET, RUN_MODE
from fcl_data_utils import get_required_split, normalize_seq_feature, normalize_stat_feature
from fcl_models_v4 import V4DualBackbone
from data_pipeline.federated_contract import validate_contract
from fcl_protocol import (
    CLIENTS,
    NUM_CLIENTS,
    NUM_TASKS,
    PROTOCOL_VERSION,
    ROUNDS_PER_TASK,
    SEED,
    TASKS,
)

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    PrivacyEngine = None
    OPACUS_AVAILABLE = False


def set_seed(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()
CLIENT_MAPPING = {i: name for i, name in enumerate(CLIENTS)}
TASK_MAPPING = {i: name for i, name in enumerate(TASKS)}


class V4Args:
    def __init__(self):
        self.num_clients = NUM_CLIENTS
        self.num_tasks = NUM_TASKS
        self.rounds_per_task = ROUNDS_PER_TASK
        self.lr = 0.001
        self.batch_size = 64
        self.local_epochs = 3
        self.use_stat = True
        self.use_seq = True
        self.use_adapter = False
        self.use_se = False
        self.use_cosface = False
        self.use_logit_adj = False
        self.use_replay = False
        self.use_dp = False
        self.replay_capacity = int(os.environ.get("V4_REPLAY_CAPACITY", "240"))
        self.replay_method = os.environ.get("V4_REPLAY_METHOD", "er").strip().lower()
        if self.replay_method not in {"er", "derpp"}:
            raise ValueError("V4_REPLAY_METHOD must be one of: er, derpp")
        self.der_alpha = float(os.environ.get("V4_DER_ALPHA", "0.5"))
        self.der_beta = float(os.environ.get("V4_DER_BETA", "0.5"))
        self.dp_noise_multiplier = float(os.environ.get("DP_NOISE_MULTIPLIER", "1.1"))
        self.dp_max_grad_norm = 1.0
        self.freeze_backbone = False
        self.frozen_trainable = os.environ.get(
            "V4_FROZEN_TRAINABLE", "adapter_classifier"
        ).strip().lower()
        if self.frozen_trainable not in {"adapter_classifier", "classifier"}:
            raise ValueError(
                "V4_FROZEN_TRAINABLE must be one of: adapter_classifier, classifier"
            )
        self.aggregate_head_only = False
        self.version = "V4"
        self.start_task = int(os.environ.get("V4_START_TASK", "0"))
        if self.start_task not in {0, 1}:
            raise ValueError("V4_START_TASK currently supports only 0 or 1")
        self.initial_checkpoint = os.environ.get("V4_INITIAL_CHECKPOINT", "").strip()
        # Optional fixed local-update budget for matched-distribution studies.
        # A value of 0 preserves the normal epoch-based training protocol.
        self.fixed_updates = int(os.environ.get("V4_FIXED_UPDATES", "0"))
        schedule = os.environ.get("V4_ROUNDS_PER_TASK", "").strip()
        if schedule:
            values = [int(item.strip()) for item in schedule.split(",") if item.strip()]
            if len(values) != self.num_tasks or any(value <= 0 for value in values):
                raise ValueError(
                    f"V4_ROUNDS_PER_TASK must contain {self.num_tasks} positive integers"
                )
            self.rounds_schedule = values
        else:
            self.rounds_schedule = [self.rounds_per_task] * self.num_tasks


def variant_name():
    return os.environ.get("V4_VARIANT", "FAIR").upper()


def stage_name():
    return os.environ.get("V4_STAGE", "full" if variant_name() == "FULL" else "fair").lower()


def component_flags():
    """Return a cumulative V4 ablation configuration.

    The stage names intentionally form a fixed order so each new run adds one
    component to the previous run. ``V4_FULL`` defaults to the complete stage.
    """
    override = os.environ.get("V4_COMPONENTS")
    if override is not None:
        selected = {item.strip().lower() for item in override.split(",") if item.strip()}
        unknown = selected - {"adapter", "se", "cosface", "replay"}
        if unknown:
            raise ValueError(f"Unknown V4_COMPONENTS={sorted(unknown)}")
        return {
            "adapter": "adapter" in selected,
            "se": "se" in selected,
            "cosface": "cosface" in selected,
            "replay": "replay" in selected,
            "dp": False,
        }
    if variant_name() == "FAIR":
        return dict(adapter=False, se=False, cosface=False, replay=False, dp=False)
    stages = {
        "base": (0, 0, 0, 0, 0),
        "adapter": (1, 0, 0, 0, 0),
        "se": (1, 1, 0, 0, 0),
        "cosface": (1, 1, 1, 0, 0),
        "replay": (1, 1, 1, 1, 0),
        "full": (1, 1, 1, 1, 1),
    }
    if stage_name() not in stages:
        raise ValueError(f"Unknown V4_STAGE={stage_name()}; use {', '.join(stages)}")
    adapter, se, cosface, replay, dp = stages[stage_name()]
    return dict(adapter=bool(adapter), se=bool(se), cosface=bool(cosface), replay=bool(replay), dp=bool(dp))


def _split(data, client_idx, task_idx, split):
    order = os.environ.get("V4_TASK_ORDER")
    if order:
        values = [item.strip() for item in order.split(",") if item.strip()]
        if len(values) != NUM_TASKS:
            raise ValueError(f"V4_TASK_ORDER must contain {NUM_TASKS} task indices")
        source_task = values[task_idx]
        if source_task.isdigit():
            source_task = f"Task_{int(source_task)}"
    else:
        source_task = TASK_MAPPING[task_idx]
    return get_required_split(data, CLIENT_MAPPING[client_idx], source_task, split)


def build_global_label_map(data_dict):
    labels = set()
    for client_idx in range(NUM_CLIENTS):
        for task_idx in range(NUM_TASKS):
            labels.update(sample["label"] for sample in _split(data_dict, client_idx, task_idx, "train"))
    return {label: idx for idx, label in enumerate(sorted(labels))}


def build_class_counts(data_dict, label_map):
    counts = [0] * len(label_map)
    for client_idx in range(NUM_CLIENTS):
        for task_idx in range(NUM_TASKS):
            for sample in _split(data_dict, client_idx, task_idx, "train"):
                counts[label_map[sample["label"]]] += 1
    return counts


def prior_mode():
    mode = os.environ.get("V4_PRIOR_MODE", "full").strip().lower()
    if mode not in {"none", "online", "full"}:
        raise ValueError("V4_PRIOR_MODE must be one of: none, online, full")
    return mode


def build_class_counts_until(data_dict, label_map, last_task):
    counts = [0] * len(label_map)
    for client_idx in range(NUM_CLIENTS):
        for task_idx in range(last_task + 1):
            for sample in _split(data_dict, client_idx, task_idx, "train"):
                counts[label_map[sample["label"]]] += 1
    return counts


def make_loader(samples, label_map, batch_size, use_seq=True, shuffle=False):
    stat = np.asarray([normalize_stat_feature(s["stat_raw"]) for s in samples], dtype=np.float32)
    labels = np.asarray([label_map[s["label"]] for s in samples], dtype=np.int64)
    if use_seq:
        seq = np.asarray([normalize_seq_feature(s["seq_feature"]) for s in samples], dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(stat), torch.from_numpy(seq), torch.from_numpy(labels))
    else:
        dataset = TensorDataset(torch.from_numpy(stat), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_global_loader(data_dict, task_idx, split, label_map, batch_size, use_seq=True):
    samples = []
    for client_idx in range(NUM_CLIENTS):
        samples.extend(_split(data_dict, client_idx, task_idx, split))
    return make_loader(samples, label_map, batch_size, use_seq, shuffle=False)


def setup_logger(path):
    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        console = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)
    return logger


def evaluate(model, loader, device):
    model.eval()
    targets, preds, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                stat, seq, target = batch
                logits = model(stat.to(device), seq.to(device))
            else:
                stat, target = batch
                logits = model(stat.to(device))
            probability = torch.softmax(logits, dim=1)
            targets.extend(target.numpy())
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            probs.extend(probability.cpu().numpy())
    y_true, y_pred, y_prob = np.asarray(targets), np.asarray(preds), np.asarray(probs)
    present = np.unique(y_true)
    return {
        "acc": 100.0 * float(np.mean(y_true == y_pred)),
        "macro_f1": 100.0 * f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "recall": 100.0 * recall_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "precision": 100.0 * precision_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "auc": auc_score(y_true, y_prob, present),
    }


def auc_score(y_true, y_prob, present):
    try:
        if len(present) == 2:
            binary_target = (y_true == present[1]).astype(np.int64)
            return 100.0 * roc_auc_score(binary_target, y_prob[:, int(present[1])])
        local_target = np.searchsorted(present, y_true)
        local_prob = y_prob[:, present]
        local_prob = local_prob / np.clip(local_prob.sum(axis=1, keepdims=True), 1e-12, None)
        return 100.0 * roc_auc_score(local_target, local_prob, multi_class="ovr", average="macro")
    except ValueError:
        return float("nan")


def aggregate(global_model, client_states, client_sizes, uploaded_keys=None):
    state = global_model.state_dict()
    keys = uploaded_keys or state.keys()
    total = float(sum(client_sizes))
    for key in keys:
        if key not in state:
            continue
        state[key] = sum(
            (client_sizes[i] / total) * client_states[i][key].to(state[key].device)
            for i in range(len(client_states))
        ).to(dtype=state[key].dtype)
    global_model.load_state_dict(state)


class V4Client:
    def __init__(self, client_id, device, args, components):
        self.client_id = client_id
        self.device = device
        self.args = args
        self.components = components
        self.replay_buffer = []
        self.epsilon = None

    def _trainable_parameters(self, model, freeze_backbone):
        for name, parameter in model.named_parameters():
            if not freeze_backbone:
                parameter.requires_grad = True
            elif self.args.frozen_trainable == "classifier":
                parameter.requires_grad = "classifier" in name
            else:
                parameter.requires_grad = "classifier" in name or "adapter" in name
        return [p for p in model.parameters() if p.requires_grad]

    def train(self, model, loader, task_id, logger):
        model.to(self.device)
        # Opacus validates the module before wrapping it and requires train mode.
        model.train(True)
        freeze = task_id > 0 and self.args.freeze_backbone
        parameters = self._trainable_parameters(model, freeze)
        optimizer = torch.optim.AdamW(parameters, lr=self.args.lr, weight_decay=1e-4)
        private_loader = loader
        privacy_engine = None
        if self.components["dp"]:
            if not OPACUS_AVAILABLE:
                raise RuntimeError("V4_FULL/full requires opacus; install opacus in zyz_env first.")
            privacy_engine = PrivacyEngine(accountant="rdp")
            model.train(True)
            logger.info(
                "DP preflight client=%d task=%d training=%s trainable=%d",
                self.client_id, task_id, model.training,
                sum(parameter.requires_grad for parameter in model.parameters()),
            )
            model, optimizer, private_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=self.args.dp_noise_multiplier,
                max_grad_norm=self.args.dp_max_grad_norm,
            )
        model.train()
        updates = 0
        replay_batches = 0
        replay_samples = 0
        for _ in range(self.args.local_epochs):
            for batch in private_loader:
                if len(batch) == 3:
                    stat, seq, target = batch
                    stat, seq = stat.to(self.device), seq.to(self.device)
                    logits = model(stat, seq, targets=target.to(self.device))
                else:
                    stat, target = batch
                    stat = stat.to(self.device)
                    logits = model(stat, targets=target.to(self.device))
                loss = F.cross_entropy(logits, target.to(self.device))
                if self.components["replay"] and self.replay_buffer:
                    replay_loader = self._replay_loader(loader, model)
                    if replay_loader is not None:
                        for replay_batch in replay_loader:
                            if self.args.replay_method == "derpp":
                                rstat, rseq, rtarget, stored_logits = replay_batch
                                rstat = rstat.to(self.device)
                                rseq = rseq.to(self.device)
                                rtarget = rtarget.to(self.device)
                                stored_logits = stored_logits.to(self.device)
                                current_logits = model(rstat, rseq)
                                margin_logits = model(rstat, rseq, targets=rtarget)
                                loss = (
                                    loss
                                    + self.args.der_alpha
                                    * F.mse_loss(current_logits, stored_logits)
                                    + self.args.der_beta
                                    * F.cross_entropy(margin_logits, rtarget)
                                )
                            else:
                                rstat, rseq, rtarget = replay_batch
                                rlogits = model(
                                    rstat.to(self.device), rseq.to(self.device),
                                    targets=rtarget.to(self.device),
                                )
                                loss = loss + 0.5 * F.cross_entropy(
                                    rlogits, rtarget.to(self.device)
                                )
                            replay_batches += 1
                            replay_samples += int(rtarget.shape[0])
                            break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates += 1
                if self.args.fixed_updates > 0 and updates >= self.args.fixed_updates:
                    break
            if self.args.fixed_updates > 0 and updates >= self.args.fixed_updates:
                break
        if privacy_engine is not None:
            self.epsilon = privacy_engine.get_epsilon(delta=1e-5)
        trained_model = model._module if hasattr(model, "_module") else model
        state = {k: v.detach().cpu().clone() for k, v in trained_model.state_dict().items()}
        resources = {
            "optimizer_updates": updates,
            "replay_batches": replay_batches,
            "replay_samples": replay_samples,
        }
        if freeze:
            if self.args.frozen_trainable == "classifier":
                upload = [key for key in state if "classifier" in key]
            else:
                upload = [
                    key for key in state if "classifier" in key or "adapter" in key
                ]
            return state, upload, resources
        return state, list(state), resources

    def _replay_loader(self, current_loader, model):
        if not self.replay_buffer:
            return None
        samples = self.replay_buffer
        labels = torch.tensor([item[2] for item in samples], dtype=torch.long)
        stat = torch.stack([item[0] for item in samples])
        seq = torch.stack([item[1] for item in samples])
        tensors = [stat, seq, labels]
        if self.args.replay_method == "derpp":
            tensors.append(torch.stack([item[3] for item in samples]))
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=min(len(samples), current_loader.batch_size or 32),
            shuffle=True,
        )

    def store_replay(self, samples, label_map, model=None):
        if not self.components["replay"]:
            return
        if self.args.replay_capacity <= 0:
            self.replay_buffer = []
            return
        candidates = []
        for sample in samples:
            candidates.append((
                torch.tensor(normalize_stat_feature(sample["stat_raw"]), dtype=torch.float32),
                torch.tensor(normalize_seq_feature(sample["seq_feature"]), dtype=torch.float32),
                label_map[sample["label"]],
            ))
        if self.args.replay_method == "derpp":
            if model is None:
                raise ValueError("DER++ replay storage requires the current global model")
            was_training = model.training
            model.eval()
            enriched = []
            with torch.no_grad():
                for start in range(0, len(candidates), self.args.batch_size):
                    batch = candidates[start:start + self.args.batch_size]
                    stat = torch.stack([item[0] for item in batch]).to(self.device)
                    seq = torch.stack([item[1] for item in batch]).to(self.device)
                    logits = model(stat, seq).detach().cpu()
                    enriched.extend(
                        (item[0], item[1], item[2], logit)
                        for item, logit in zip(batch, logits)
                    )
            model.train(was_training)
            candidates = enriched
        self.replay_buffer.extend(candidates)
        by_class = {}
        for item in self.replay_buffer:
            by_class.setdefault(item[2], []).append(item)
        per_class = max(1, self.args.replay_capacity // max(1, len(by_class)))
        self.replay_buffer = [item for values in by_class.values() for item in values[-per_class:]]
        self.replay_buffer = self.replay_buffer[-self.args.replay_capacity:]


def build_model(args, label_map, class_counts, components, device):
    model = V4DualBackbone(
        stat_dim=CFG["stat_dim"], seq_dim=512, hidden_dim=getattr(args, "hidden_dim", 128),
        num_classes=len(label_map), use_stat=args.use_stat, use_seq=args.use_seq,
        use_adapter=components["adapter"], use_se=components["se"],
        use_cosface=components["cosface"], use_logit_adj=prior_mode() != "none",
        class_counts=class_counts,
    )
    return model.to(device)


def run_fcl_pipeline(args, data, label_map, device, logger, eval_split="val"):
    components = component_flags()
    args.version = f"V4_{variant_name()}_{stage_name()}"
    args.use_adapter = components["adapter"]
    args.use_se = components["se"]
    args.use_cosface = components["cosface"]
    args.use_replay = components["replay"]
    args.use_dp = components["dp"]
    freeze_override = os.environ.get("V4_FREEZE_BACKBONE")
    if freeze_override is None:
        args.freeze_backbone = components["adapter"] and variant_name() == "FULL"
    else:
        args.freeze_backbone = freeze_override.strip().lower() in {"1", "true", "yes"}
    args.aggregate_head_only = args.freeze_backbone
    mode = prior_mode()
    class_counts = build_class_counts(data, label_map)
    if mode == "none":
        class_counts = [1] * len(label_map)
    elif mode == "online":
        class_counts = build_class_counts_until(data, label_map, 0)
    global_model = build_model(args, label_map, class_counts, components, device)
    clients = [V4Client(i, device, args, components) for i in range(args.num_clients)]
    matrices = {name: np.zeros((args.num_tasks, args.num_tasks)) for name in ("acc", "macro_f1", "recall", "precision", "auc")}
    privacy = []
    task_resources = []
    task1_round_history = []
    record_task1_curve = os.environ.get("V4_RECORD_TASK1_CURVE", "0").strip().lower() in {
        "1", "true", "yes"
    }

    if args.start_task:
        if not args.initial_checkpoint:
            raise ValueError("V4_INITIAL_CHECKPOINT is required when V4_START_TASK=1")
        checkpoint = torch.load(args.initial_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        global_model.load_state_dict(checkpoint)
        logger.info(
            "V4 resumed start_task=%d checkpoint=%s frozen_trainable=%s",
            args.start_task + 1, args.initial_checkpoint, args.frozen_trainable,
        )
        if components["replay"]:
            for previous_task in range(args.start_task):
                for client_idx, client in enumerate(clients):
                    client.store_replay(
                        _split(data, client_idx, previous_task, "train"),
                        label_map,
                        global_model,
                    )
        for previous_task in range(args.start_task):
            metrics = evaluate(
                global_model,
                make_global_loader(
                    data, previous_task, eval_split, label_map,
                    args.batch_size, args.use_seq,
                ),
                device,
            )
            for key in matrices:
                matrices[key][previous_task, previous_task] = metrics[key]

    for task_id in range(args.start_task, args.num_tasks):
        task_started = time.monotonic()
        if mode == "online":
            global_model.set_class_counts(build_class_counts_until(data, label_map, task_id))
        active_clients = []
        for client_idx, client in enumerate(clients):
            samples = _split(data, client_idx, task_id, "train")
            if samples:
                active_clients.append(
                    (client_idx, client, make_loader(samples, label_map, args.batch_size, args.use_seq, True))
                )
        if not active_clients:
            raise ValueError(f"Task_{task_id + 1} has no training samples across all clients")
        loaders = [entry[2] for entry in active_clients]
        sizes = [len(loader.dataset) for loader in loaders]
        uploaded_keys = None
        rounds_this_task = args.rounds_schedule[task_id]
        resource = {
            "task": task_id + 1,
            "backbone_frozen_during_task": bool(task_id > 0 and args.freeze_backbone),
            "rounds": rounds_this_task,
            "client_rounds": 0,
            "optimizer_updates": 0,
            "replay_batches": 0,
            "replay_samples": 0,
            "upload_bytes": 0,
        }
        logger.info(
            "V4 schedule task=%d rounds=%d local_epochs=%d",
            task_id + 1, rounds_this_task, args.local_epochs,
        )
        for round_id in range(rounds_this_task):
            client_states = []
            round_keys = []
            for _, client, loader in active_clients:
                local = copy.deepcopy(global_model)
                # Evaluation switches the global model to eval mode; every local
                # copy must be returned to train mode before DP validation.
                local.train(True)
                state, keys, train_resources = client.train(local, loader, task_id, logger)
                client_states.append(state)
                round_keys.append(keys)
                resource["client_rounds"] += 1
                for key in ("optimizer_updates", "replay_batches", "replay_samples"):
                    resource[key] += int(train_resources[key])
                if client.epsilon is not None:
                    privacy.append(client.epsilon)
            uploaded_keys = sorted(set(round_keys[0]).intersection(*[set(keys) for keys in round_keys[1:]]))
            aggregate(global_model, client_states, sizes, uploaded_keys)
            payload_sizes = [
                len(pickle.dumps({key: state[key] for key in uploaded_keys}))
                for state in client_states
            ]
            resource["upload_bytes"] += int(sum(payload_sizes))
            if round_id == 0:
                payload = np.mean(payload_sizes) / 1024.0
                logger.info(
                    "V4 payload task=%d round=%d active_clients=%d avg_kb=%.2f",
                    task_id, round_id, len(active_clients), payload,
                )
            if record_task1_curve and task_id == 0:
                row = {"round": round_id + 1}
                for split_name in ("val", "test"):
                    round_metrics = evaluate(
                        global_model,
                        make_global_loader(
                            data, task_id, split_name, label_map, args.batch_size, args.use_seq
                        ),
                        device,
                    )
                    for metric_name, value in round_metrics.items():
                        row[f"{split_name}_{metric_name}"] = value
                task1_round_history.append(row)
                logger.info(
                    "V4 task1 curve round=%d val_macro_f1=%.4f test_macro_f1=%.4f",
                    round_id + 1, row["val_macro_f1"], row["test_macro_f1"],
                )

        task_resources.append(resource)

        if components["replay"]:
            for client_idx, client in enumerate(clients):
                client.store_replay(
                    _split(data, client_idx, task_id, "train"), label_map, global_model
                )
        for past_task in range(task_id + 1):
            metrics = evaluate(
                global_model,
                make_global_loader(data, past_task, eval_split, label_map, args.batch_size, args.use_seq),
                device,
            )
            for key in matrices:
                matrices[key][task_id, past_task] = metrics[key]

        checkpoint_root = os.environ.get("V4_TASK_CHECKPOINT_ROOT")
        if checkpoint_root:
            os.makedirs(checkpoint_root, exist_ok=True)
            torch.save(
                {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()},
                os.path.join(checkpoint_root, f"task_{task_id + 1}.pth"),
            )
        callback = getattr(args, "freeze_policy", None)
        if callback is not None:
            args.freeze_backbone = bool(callback(global_model, data, label_map, task_id, device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        resource["task_wall_seconds_including_eval_and_policy"] = time.monotonic() - task_started

    last = args.num_tasks - 1
    bwt = float(np.mean([matrices["macro_f1"][last, i] - matrices["macro_f1"][i, i] for i in range(last)])) if last else 0.0
    forgetting_values = []
    for task_idx in range(last):
        best_seen = max(matrices["macro_f1"][step, task_idx] for step in range(task_idx, last + 1))
        forgetting_values.append(best_seen - matrices["macro_f1"][last, task_idx])
    forgetting = float(np.mean(forgetting_values)) if forgetting_values else 0.0
    auc_values = matrices["auc"][last]
    valid_auc = auc_values[~np.isnan(auc_values)]
    result = {
        f"{eval_split}_acc": round(float(np.mean(matrices["acc"][last])), 4),
        f"{eval_split}_macro_f1": round(float(np.mean(matrices["macro_f1"][last])), 4),
        f"{eval_split}_recall": round(float(np.mean(matrices["recall"][last])), 4),
        f"{eval_split}_precision": round(float(np.mean(matrices["precision"][last])), 4),
        f"{eval_split}_auc": round(float(np.mean(valid_auc)), 4) if len(valid_auc) else None,
        f"{eval_split}_f1_bwt": round(bwt, 4),
        f"{eval_split}_forgetting": round(forgetting, 4),
        "model_version": args.version,
        "variant": variant_name(),
        "stage": stage_name(),
        "components": components,
        "protocol_version": PROTOCOL_VERSION,
        "eval_split": eval_split,
        "epsilon": round(max(privacy), 4) if privacy else None,
        "label_map": label_map,
        "prior_mode": mode,
        "freeze_backbone": bool(args.freeze_backbone),
        "fixed_updates": int(args.fixed_updates),
        "start_task": int(args.start_task),
        "initial_checkpoint": args.initial_checkpoint or None,
        "frozen_trainable": args.frozen_trainable,
        "replay_method": args.replay_method,
        "der_alpha": args.der_alpha if args.replay_method == "derpp" else None,
        "der_beta": args.der_beta if args.replay_method == "derpp" else None,
        "rounds_schedule": list(args.rounds_schedule),
        "hidden_dim": getattr(args, "hidden_dim", 128),
        "training_resources": {
            "by_task": task_resources,
            "total_client_rounds": int(sum(item["client_rounds"] for item in task_resources)),
            "total_optimizer_updates": int(sum(item["optimizer_updates"] for item in task_resources)),
            "total_replay_batches": int(sum(item["replay_batches"] for item in task_resources)),
            "total_replay_samples": int(sum(item["replay_samples"] for item in task_resources)),
            "total_upload_bytes": int(sum(item["upload_bytes"] for item in task_resources)),
        },
        "task1_round_history": task1_round_history,
        "task_metric_matrices": {key: matrices[key].tolist() for key in matrices},
    }
    return result, {k: v.cpu() for k, v in global_model.state_dict().items()}


class Tracker:
    def __init__(self, output_root):
        self.output_root = output_root
        self.items = []
        self.model_dir = os.path.join(output_root, "top5_models")
        os.makedirs(self.model_dir, exist_ok=True)

    def update(self, score, state, info):
        self.items.append((float(score), info))
        self.items.sort(key=lambda pair: pair[0], reverse=True)
        self.items = self.items[:5]
        torch.save(state, os.path.join(self.model_dir, f"top_model_{info['trial_id']}_score_{score:.2f}.pth"))

    def export(self):
        path = os.path.join(self.output_root, "top5_models_results.csv")
        if not self.items:
            return path
        fields = ["Comprehensive_Score"] + list(self.items[0][1].keys())
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for score, info in self.items:
                writer.writerow(dict(info, Comprehensive_Score=round(score, 4)))
        return path


def optuna_objective(trial, base_args, data, label_map, device, logger, tracker):
    args = copy.deepcopy(base_args)
    args.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    args.local_epochs = trial.suggest_int("local_epochs", 2, 4)
    metrics, state = run_fcl_pipeline(args, data, label_map, device, logger, eval_split="val")
    score = metrics["val_macro_f1"]
    info = {
        "trial_id": trial.number,
        "model_version": metrics["model_version"],
        "selection_metric": "val_macro_f1",
        "val_Macro_F1": metrics["val_macro_f1"],
        "val_F1_BWT": metrics["val_f1_bwt"],
        "val_ACC": metrics["val_acc"],
        "val_REC": metrics["val_recall"],
        "val_PRE": metrics["val_precision"],
        "val_AUC": metrics["val_auc"],
        "epsilon": metrics["epsilon"],
        "stage": metrics["stage"],
    }
    tracker.update(score, state, info)
    logger.info("[%s] trial=%d val_macro_f1=%.4f", metrics["model_version"], trial.number, score)
    return score


def run_final_test(args, data, label_map, device, logger):
    return run_fcl_pipeline(args, data, label_map, device, logger, eval_split="test")


def load_data_and_labels():
    # Validate semantic identity/content before trusting a same-width new dataset.
    validate_contract(CFG["fcl_data_path"], CFG, CURRENT_DATASET)
    with open(CFG["fcl_data_path"], "rb") as handle:
        data = pickle.load(handle)
    validate_contract(CFG["fcl_data_path"], CFG, CURRENT_DATASET, data)
    return data, build_global_label_map(data)


def output_root():
    base = os.environ.get("OUTPUT_ROOT", os.path.join("experiment_logs", f"{CURRENT_DATASET}_V4_{variant_name()}"))
    if variant_name() == "FULL":
        return os.path.join(base, stage_name())
    return base


def main():
    import optuna

    root = output_root()
    os.makedirs(root, exist_ok=True)
    logger = setup_logger(os.path.join(root, f"v4_{variant_name().lower()}_{stage_name()}.log"))
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, label_map = load_data_and_labels()
    args = V4Args()
    logger.info("V4 start version=%s dataset=%s stage=%s device=%s", args.version, CURRENT_DATASET, stage_name(), device)
    logger.info("data=%s labels=%s components=%s", CFG["fcl_data_path"], label_map, component_flags())
    tracker = Tracker(root)
    study_name = f"fcl_V4_{variant_name()}_{CURRENT_DATASET}_{stage_name()}_{PROTOCOL_VERSION}"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{os.path.join(root, 'optuna.db')}",
        direction="maximize",
        load_if_exists=True,
    )
    study.set_user_attr("model_version", f"V4_{variant_name()}")
    study.set_user_attr("stage", stage_name())
    study.set_user_attr("components", json.dumps(component_flags(), sort_keys=True))
    n_trials = int(os.environ.get("N_TRIALS", "40"))
    study.optimize(lambda trial: optuna_objective(trial, args, data, label_map, device, logger, tracker), n_trials=n_trials, gc_after_trial=True)
    csv_path = tracker.export()
    with open(os.path.join(root, "best_params.json"), "w", encoding="utf-8") as handle:
        json.dump({"model_version": f"V4_{variant_name()}", "stage": stage_name(), "params": study.best_params}, handle, indent=2)
    logger.info("best_value=%.4f results=%s", study.best_value, csv_path)


if __name__ == "__main__":
    main()
