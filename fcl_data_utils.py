import hashlib
from collections import Counter

import numpy as np

from fcl_protocol import SEED, SPLITS, TRAIN_RATIO, VAL_RATIO


def stable_seed(base_seed, *parts):
    text = "::".join(str(part) for part in (base_seed,) + parts)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def split_samples(samples, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and leave room for test.")

    samples = list(samples)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    shuffled = [samples[i] for i in indices]

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def finalize_fcl_splits(fcl_data, seed=SEED, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """Convert data[client][task] lists into explicit train/val/test dictionaries."""
    split_data = {}
    for client, tasks in fcl_data.items():
        split_data[client] = {}
        for task, task_value in tasks.items():
            if is_split_task(task_value):
                split_data[client][task] = task_value
                continue
            split_seed = stable_seed(seed, client, task)
            split_data[client][task] = split_samples(
                task_value,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=split_seed,
            )
    return split_data


def is_split_task(task_value):
    return isinstance(task_value, dict) and set(SPLITS).issubset(task_value.keys())


def get_required_split(data_dict, client_key, task_key, split):
    if split not in SPLITS:
        raise ValueError(f"Invalid split '{split}'. Expected one of {SPLITS}.")
    if client_key not in data_dict:
        raise ValueError(f"Missing client key: {client_key}")
    if task_key not in data_dict[client_key]:
        raise ValueError(f"Missing task key: {task_key} for client {client_key}")
    task_value = data_dict[client_key][task_key]
    if not is_split_task(task_value):
        raise ValueError(
            f"{client_key}/{task_key} must be a dict with train/val/test splits. "
            "Regenerate the dataset with the updated splitter."
        )
    split_samples_list = task_value[split]
    if not isinstance(split_samples_list, list):
        raise ValueError(f"{client_key}/{task_key}/{split} must be a list.")
    return split_samples_list


def normalize_seq_feature(seq_feature, seq_len=512):
    arr = np.asarray(seq_feature, dtype=np.float32).reshape(-1)
    if arr.shape[0] >= seq_len:
        return arr[:seq_len]
    padded = np.zeros(seq_len, dtype=np.float32)
    padded[:arr.shape[0]] = arr
    return padded


def normalize_stat_feature(stat_feature):
    return np.asarray(stat_feature, dtype=np.float32).reshape(-1)


def sample_signature(sample):
    digest = hashlib.md5()
    digest.update(str(sample.get("label", "")).encode("utf-8"))
    for key in ("stat_raw", "seq_feature"):
        if key in sample:
            arr = np.asarray(sample[key])
            digest.update(str(arr.shape).encode("utf-8"))
            digest.update(arr.tobytes())
    return digest.hexdigest()


def label_distribution(samples):
    return Counter(sample.get("label") for sample in samples)


def count_nested_samples(data_dict):
    total = 0
    for tasks in data_dict.values():
        for task_value in tasks.values():
            if is_split_task(task_value):
                total += sum(len(task_value[split]) for split in SPLITS)
            else:
                total += len(task_value)
    return total

