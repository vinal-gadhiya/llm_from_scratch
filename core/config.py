import torch
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent

INPUT_FILE_PATH = CORE_DIR / "input.txt"
VOCAB_PATH = CORE_DIR / "vocab.json"
MERGE_PATH = CORE_DIR / "merge.txt"
CHECKPOINT_DIR = CORE_DIR / "checkpoints"

VOCAB_SIZE = 1064
D_MODEL = 512
N_HEADS = 8
HIDDEN_LAYER_DIM = 1024
N_BLOCKS = 16
SEQ_LEN = 1000
N_TRAINING_ITERATIONS = 10000
CHECKPOINT_INTERVAL = 1000
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")