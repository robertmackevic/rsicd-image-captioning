from argparse import Namespace
from os import listdir, makedirs
from typing import Optional, Dict, Tuple

import torch
from nltk.translate.bleu_score import corpus_bleu
from torch import Tensor, no_grad
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.data.tokenizer import Tokenizer
from src.metrics import AverageMeter, compute_topk_accuracy
from src.modules.model import Image2Text
from src.paths import RUNS_DIR
from src.utils import get_available_device, get_logger, save_config, save_weights


class Trainer:
    def __init__(self, config: Namespace, tokenizer: Tokenizer) -> None:
        self.config = config
        self.device = get_available_device()
        self.logger = get_logger(__name__)

        self.tokenizer = tokenizer
        self.model = Image2Text(config, vocab_size=len(tokenizer.vocab)).to(self.device)

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

        best_score = 0
        best_score_metric = self.config.best_score_metric

        for epoch in range(1, self.config.epochs + 1):
            metrics = self._train_for_epoch(train_dl)
            self.logger.info(f"[Epoch {epoch} / {self.config.epochs}]")
            self.log_metrics(metrics, summary_writer_train, epoch=epoch)

            if epoch % self.config.eval_interval == 0:
                self.logger.info("Evaluating...")
                metrics = self.eval(val_dl)
                self.log_metrics(metrics, summary_writer_eval, epoch=epoch)

                score = metrics[best_score_metric].avg

                if score > best_score:
                    best_score = score
                    self.logger.info(f"Saving best weights with {best_score_metric}: {score:.3f}")
                    save_weights(model_dir / "weights_best.pth", self.model)

            if epoch % self.config.save_interval == 0:
                self.logger.info(f"Saving model weights at epoch: {epoch}")
                save_weights(model_dir / f"weights_{epoch}.pth", self.model)

    def _train_for_epoch(self, dataloader: DataLoader) -> Dict[str, AverageMeter]:
        self.model.train()
        metrics = {
            "loss": AverageMeter(),
            "top5_accuracy": AverageMeter(),
        }

        for batch in tqdm(dataloader):
            _, decode_lengths, _, loss, top5_accuracy = self._forward(batch)

            self.decoder_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            loss.backward()

            clip_grad_value_(self.model.parameters(), self.config.grad_clip)

            self.decoder_optimizer.step()
            self.encoder_optimizer.step()

            metrics["loss"].update(loss.item(), n=sum(decode_lengths))
            metrics["top5_accuracy"].update(top5_accuracy, n=sum(decode_lengths))

        return metrics

    def eval(self, dataloader: DataLoader) -> Dict[str, AverageMeter]:
        self.model.eval()
        metrics = {
            "loss": AverageMeter(),
            "top5_accuracy": AverageMeter(),
            "bleu": AverageMeter()
        }

        for batch in tqdm(dataloader):
            with no_grad():
                predictions, decode_lengths, sort_idx, loss, top5_accuracy = self._forward(batch)

            metrics["loss"].update(loss.item(), n=sum(decode_lengths))
            metrics["top5_accuracy"].update(top5_accuracy, n=sum(decode_lengths))

            all_captions = batch[3][sort_idx.cpu()]
            metrics["bleu"].update(
                corpus_bleu(
                    list_of_references=[
                        [self.tokenizer.decode(reference) for reference in all_captions[batch_idx].tolist()]
                        for batch_idx in range(all_captions.size(0))
                    ],
                    hypotheses=[
                        self.tokenizer.decode(hypothesis)
                        for hypothesis in torch.max(predictions, dim=2).indices.tolist()
                    ]
                ),
            )

        return metrics

    def _forward(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
        images = batch[0].to(self.device)
        captions = batch[1].to(self.device)
        caption_lengths = batch[2].to(self.device)

        predictions, sorted_captions, decode_lengths, alphas, sort_idx = (
            self.model(images, captions, caption_lengths)
        )

        packed_predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True)
        packed_targets = pack_padded_sequence(sorted_captions[:, 1:], decode_lengths, batch_first=True)

        loss = self.loss_fn(packed_predictions.data, packed_targets.data)

        # Add doubly stochastic attention regularization
        loss += self.config.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        top5_accuracy = compute_topk_accuracy(packed_predictions.data, packed_targets.data, k=5)

        return predictions, decode_lengths, sort_idx, loss, top5_accuracy

    def log_metrics(
            self,
            metrics: Dict[str, AverageMeter],
            summary_writer: Optional[SummaryWriter] = None,
            epoch: Optional[int] = None
    ) -> None:
        message = ""
        for metric, value in metrics.items():
            message += f"{metric}: {value.avg:.3f} | "

            if epoch is not None and summary_writer is not None:
                summary_writer.add_scalar(tag=metric, scalar_value=value.avg, global_step=epoch)

        self.logger.info(message[:-3])
