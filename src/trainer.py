import statistics
from argparse import Namespace
from os import listdir, makedirs
from typing import Optional, Dict, Tuple

import nltk
import torch
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from torch import Tensor, no_grad
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_value_
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.data.tokenizer import Tokenizer
from src.modules.model import Image2Text
from src.paths import RUNS_DIR
from src.utils import get_available_device, get_logger, save_config, save_weights


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    BLEU_WEIGHTS = [
        (1.0,),
        (0.5, 0.5),
        (0.333, 0.333, 0.334),
        (0.25, 0.25, 0.25, 0.25),
    ]

    def __init__(self, config: Namespace, tokenizer: Tokenizer) -> None:
        nltk.download("wordnet", quiet=True)

        self.config = config
        self.device = get_available_device()
        self.logger = get_logger(__name__)

        self.tokenizer = tokenizer
        self.model = Image2Text(config, vocab_size=len(tokenizer.vocab)).to(self.device)

        if self.config.finetune_encoder:
            self.encoder_optimizer = Adam(
                params=filter(lambda p: p.requires_grad, self.model.encoder.parameters()),
                lr=self.config.encoder_lr
            )

        self.decoder_optimizer = Adam(
            params=self.model.decoder.parameters(),
            lr=self.config.decoder_lr
        )

        self.loss_fn = CrossEntropyLoss().to(self.device)
        self.rouge = Rouge(metrics=["rouge-l"], stats=["f"])

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
        }

        for batch in tqdm(dataloader):
            loss, _, _ = self._forward(batch)

            self.decoder_optimizer.zero_grad()
            if self.config.finetune_encoder:
                self.encoder_optimizer.zero_grad()

            loss.backward()

            clip_grad_value_(self.model.parameters(), self.config.grad_clip)

            self.decoder_optimizer.step()
            if self.config.finetune_encoder:
                self.encoder_optimizer.step()

            metrics["loss"].update(loss.item())

        return metrics

    def eval(self, dataloader: DataLoader) -> Dict[str, AverageMeter]:
        self.model.eval()
        metrics = {
            "loss": AverageMeter(),
            "metric/bleu1": AverageMeter(),
            "metric/bleu2": AverageMeter(),
            "metric/bleu3": AverageMeter(),
            "metric/bleu4": AverageMeter(),
            "metric/meteor": AverageMeter(),
            "metric/rouge-l": AverageMeter(),
        }

        for batch in tqdm(dataloader):
            with no_grad():
                loss, predictions, sort_idx = self._forward(batch)

            all_captions = batch[3][sort_idx.cpu()] if self.model.decoder_type == "lstm" else batch[3]
            metrics["loss"].update(loss.item())

            list_of_references = [
                [self.tokenizer.decode(reference) for reference in all_captions[batch_idx].tolist()]
                for batch_idx in range(all_captions.size(0))
            ]
            hypotheses = [
                self.tokenizer.decode(hypothesis)
                for hypothesis in torch.max(predictions, dim=2).indices.tolist()
            ]

            bleu1, bleu2, bleu3, bleu4 = corpus_bleu(
                list_of_references=list_of_references,
                hypotheses=hypotheses,
                weights=self.BLEU_WEIGHTS,
            )

            meteor = statistics.mean(
                meteor_score([reference.split() for reference in list_of_references[idx]], hypothesis.split())
                for idx, hypothesis in enumerate(hypotheses)
            )

            rouge_l = statistics.mean(
                self.rouge.get_scores(
                    hypotheses,
                    [references[i] for references in list_of_references],
                    avg=True,
                )["rouge-l"]["f"]
                for i in range(all_captions.size(1))
            )

            metrics["metric/bleu1"].update(bleu1)
            metrics["metric/bleu2"].update(bleu2)
            metrics["metric/bleu3"].update(bleu3)
            metrics["metric/bleu4"].update(bleu4)
            metrics["metric/meteor"].update(meteor)
            metrics["metric/rouge-l"].update(rouge_l)

        return metrics

    def _forward(self, batch: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        images = batch[0].to(self.device)
        captions = batch[1].to(self.device)

        loss, predictions, targets, alphas, sort_idx = None, None, None, None, None

        if self.model.decoder_type == "lstm":
            caption_lengths = batch[2].to(self.device)

            predictions, sorted_captions, alphas, sort_idx = (
                self.model(images, captions, caption_lengths)
            )
            targets = sorted_captions[:, 1:]

        elif self.model.decoder_type == "transformer":
            predictions = self.model(images, captions[:, :-1])
            targets = captions[:, 1:]

        else:
            raise ValueError(f"Unknown decoder type: `{self.model.decoder_type}`")

        loss = self.loss_fn(predictions.permute(0, 2, 1), targets)

        if self.model.decoder_type == "lstm":
            # Add doubly stochastic attention regularization
            loss += self.model.decoder.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        return loss, predictions, sort_idx

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
