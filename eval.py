from argparse import Namespace, ArgumentParser
from pathlib import Path

from torch.utils.data.dataloader import DataLoader

from src.data.collate import collate_fn
from src.data.dataset import RSICDDataset
from src.data.tokenizer import Tokenizer
from src.paths import DATASET_DIR, RUNS_DIR
from src.trainer import Trainer
from src.utils import (
    get_logger,
    load_config,
    seed_everything,
    extract_rsicd_dataset,
    load_weights,
    count_parameters
)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-v", "--version", type=str, required=True, help="v1, v2, v3, etc.")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Name of the .pth file containing weights")
    parser.add_argument("-s", "--split", type=str, required=True, help=r"train, val or test")
    parser.add_argument("--zip-file", type=Path, required=False, default=Path("RSICD.zip"))
    return parser.parse_args()


def run(version: str, weights: str, split: str, zip_file: Path) -> None:
    logger = get_logger(__name__)

    if not DATASET_DIR.is_dir():
        extract_rsicd_dataset(zip_file)

    model_dir = RUNS_DIR / version
    logger.info(f"Loading objects from `{model_dir}`...")

    config = load_config(filepath=model_dir / "config.json")
    tokenizer = Tokenizer.init_from_file(filepath=model_dir / "tokenizer.json")

    seed_everything(config.seed)

    logger.info(f"Preparing the data...")
    dataset = RSICDDataset(config, tokenizer, split)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    trainer = Trainer(config, tokenizer)
    trainer.model = load_weights(filepath=model_dir / weights, model=trainer.model)
    logger.info(f"Number of model parameters: {count_parameters(trainer.model)}")

    try:
        trainer.log_metrics(trainer.eval(dataloader))
    except KeyboardInterrupt:
        logger.info("Evaluation terminated.")


if __name__ == "__main__":
    run(**vars(parse_args()))
