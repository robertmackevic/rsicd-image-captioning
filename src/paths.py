from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATASET_DIR = DATA_DIR / "RSICD"
RUNS_DIR = ROOT_DIR / ".runs"

CONFIG_FILE = ROOT_DIR / "config.json"
TOKENIZER_FILE = DATA_DIR / "tokenizer.json"
