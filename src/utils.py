import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

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


def count_parameters(module: Module) -> int:
    return sum(p.numel() for p in module.parameters())


def load_config() -> Namespace:
    with open(CONFIG_FILE, "r") as config:
        return Namespace(**json.load(config))


def save_config(config: Namespace, filepath: Path) -> None:
    with open(filepath, "w") as file:
        json.dump(vars(config), file, indent=4)
