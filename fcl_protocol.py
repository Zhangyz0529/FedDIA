import os


SEED = int(os.environ.get("SEED", 42))
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 3))
NUM_TASKS = int(os.environ.get("NUM_TASKS", 3))
ROUNDS_PER_TASK = int(os.environ.get("ROUNDS_PER_TASK", 5))
LOCAL_EPOCHS_DEFAULT = int(os.environ.get("LOCAL_EPOCHS", 3))

PROTOCOL_VERSION = os.environ.get(
    "FCL_PROTOCOL_VERSION",
    f"p4_fcl_v1_train_val_test_r{ROUNDS_PER_TASK}",
)

SPLITS = ("train", "val", "test")
SELECTION_SPLIT = "val"
FINAL_EVAL_SPLIT = "test"

CLIENTS = tuple(f"Client_{chr(65 + i)}" for i in range(NUM_CLIENTS))
TASKS = tuple(f"Task_{i + 1}" for i in range(NUM_TASKS))

TRAIN_RATIO = float(os.environ.get("FCL_TRAIN_RATIO", 0.7))
VAL_RATIO = float(os.environ.get("FCL_VAL_RATIO", 0.1))
