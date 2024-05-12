from argparse import Namespace, ArgumentParser
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from torchrs.datasets import RSICD

from src.data.collate import collate_fn
from src.data.dataset import RSICDDataset
from src.data.tokenizer import Tokenizer
from src.paths import DATASET_DIR, TOKENIZER_FILE
from src.trainer import Trainer
from src.utils import (
    get_logger,
    load_config,
    seed_everything,
    extract_rsicd_dataset,
    count_parameters
)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--zip-file", type=Path, required=False, default=Path("RSICD.zip"))
    return parser.parse_args()


def run(zip_file: Path) -> None:
    logger = get_logger(__name__)
    config = load_config()
    seed_everything(config.seed)

    if not DATASET_DIR.is_dir():
        extract_rsicd_dataset(zip_file)

    if TOKENIZER_FILE.is_file():
        logger.info(f"Loading tokenizer from `{TOKENIZER_FILE}`...")
        tokenizer = Tokenizer.init_from_file()
    else:
        logger.info(f"Tokenizer file not found. Initializing tokenizer from training data...")
        train_data = RSICD(DATASET_DIR, split="train")
        tokenizer = Tokenizer.init_from_data(train_data)
        tokenizer.save(TOKENIZER_FILE)

    logger.info(f"Preparing the data...")
    train_dataset = RSICDDataset(config, tokenizer, split="train")
    val_dataset = RSICDDataset(config, tokenizer, split="val")

    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dl = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    trainer = Trainer(config, tokenizer)
    logger.info(f"Number of trainable parameters: {count_parameters(trainer.model, trainable=True)}")

    try:
        trainer.fit(train_dl, val_dl)
    except KeyboardInterrupt:
        logger.info("Training terminated.")


if __name__ == "__main__":
    run(**vars(parse_args()))
