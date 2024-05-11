import json
import logging
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import Module

from src.paths import DATASET_DIR, CONFIG_FILE


def extract_rsicd_dataset(zip_path: Path, override: bool = False) -> None:
    if DATASET_DIR.is_dir() and not override:
        print("Dataset is already extracted. Set `override=True` to extract again.")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("Unzipping...")
    DATASET_DIR.mkdir(exist_ok=True)
    with ZipFile(zip_path, "r") as file:
        file.extractall(DATASET_DIR)

    print("Dataset extracted.")


def get_available_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(name)
    return logger


def count_parameters(module: Module, trainable: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if not trainable or (trainable and p.requires_grad))


def load_config(filepath: Path = CONFIG_FILE) -> Namespace:
    with open(filepath, "r") as config:
        return Namespace(**json.load(config))


def save_config(config: Namespace, filepath: Path) -> None:
    with open(filepath, "w") as file:
        json.dump(vars(config), file, indent=4)


def load_weights(filepath: Path, model: Module) -> Module:
    checkpoint = torch.load(filepath, map_location=get_available_device())
    model.load_state_dict(checkpoint)
    return model


def save_weights(filepath: Path, model: Module) -> None:
    torch.save(model.state_dict(), filepath)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
