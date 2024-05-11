import statistics
from argparse import Namespace
from os import listdir, makedirs
from typing import List, Optional

from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.data.tokenizer import Tokenizer
from src.modules.model import EncoderDecoderCaptioning
from src.paths import RUNS_DIR
from src.utils import get_available_device, get_logger, save_config, save_weights


class Trainer:
    def __init__(self, config: Namespace, tokenizer: Tokenizer) -> None:
        self.config = config
        self.device = get_available_device()
        self.logger = get_logger(__name__)

        self.tokenizer = tokenizer
        self.model = EncoderDecoderCaptioning(config).to(self.device)

        self.encoder_optimizer = Adam(
            params=filter(lambda p: p.requires_grad, self.model.encoder.parameters()),
            lr=self.config.encoder_lr
        )
        self.decoder_optimizer = Adam(
            params=self.model.decoder.parameters(),
            lr=self.config.decoder_lr
        )

        self.loss_fn = CrossEntropyLoss().to(self.device)

    def fit(self, train_dl: DataLoader, val_dl: DataLoader) -> None:
        RUNS_DIR.mkdir(exist_ok=True)
        model_dir = RUNS_DIR / f"v{len(listdir(RUNS_DIR)) + 1}"

        summary_writer_train = SummaryWriter(log_dir=str(model_dir / "train"))
        summary_writer_eval = SummaryWriter(log_dir=str(model_dir / "eval"))

        makedirs(summary_writer_train.log_dir, exist_ok=True)
        makedirs(summary_writer_eval.log_dir, exist_ok=True)
        save_config(self.config, model_dir / "config.json")
        self.tokenizer.save(model_dir / "tokenizer.json")

        for epoch in range(1, self.config.epochs + 1):
            losses = self._train_for_epoch(train_dl)
            self.logger.info(f"[Epoch {epoch} / {self.config.epochs}]")
            self.log_losses(losses, summary_writer_train, epoch=epoch)

            if epoch % self.config.eval_interval == 0:
                self.logger.info("Evaluating...")
                self.eval(val_dl)

            if epoch % self.config.save_interval == 0:
                self.logger.info(f"Saving model weights at epoch: {epoch}")
                save_weights(model_dir / f"weights_{epoch}.pth", self.model)

    def _train_for_epoch(self, dataloader: DataLoader) -> List[float]:
        self.model.train()
        losses = []

        for batch in tqdm(dataloader):
            images = batch[0].to(self.device)
            all_captions = batch[1].to(self.device)
            all_caption_lengths = batch[2].to(self.device)

            captions = all_captions[:, 0, :]
            caption_length = all_caption_lengths[:, 0].unsqueeze(1)
            predictions, sorted_captions, decode_lengths, alphas, _ = self.model(images, captions, caption_length)

            targets = sorted_captions[:, 1:]
            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            loss = self.loss_fn(predictions.data, targets.data)

            # Add doubly stochastic attention regularization
            loss += self.config.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            self.decoder_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            loss.backward()

            clip_grad_value_(self.model.parameters(), self.config.grad_clip)

            self.decoder_optimizer.step()
            self.encoder_optimizer.step()

            losses.append(loss.item())

        return losses

    def eval(self, dataloader: DataLoader):
        pass

    def log_losses(self, losses: List[float], summary_writer: SummaryWriter, epoch: Optional[int] = None) -> None:
        loss = statistics.mean(losses)

        if epoch is not None:
            summary_writer.add_scalar(tag="loss", scalar_value=loss, global_step=epoch)

        self.logger.info(f"loss: {loss:.3f}")
