from pathlib import Path
from zipfile import ZipFile

from src.paths import DATASET_DIR


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
