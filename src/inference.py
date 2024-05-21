from io import BytesIO
from pathlib import Path
from typing import Tuple, List

import numpy as np
import requests
import skimage
from PIL import Image
from matplotlib import pyplot as plt
from torch import LongTensor, no_grad, Tensor, log_softmax
from torch.nn import Module
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ToPILImage

from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab
from src.modules.model import Image2Text
from src.paths import CONFIG_FILE, TOKENIZER_FILE
from src.utils import load_config, get_available_device, load_weights, seed_everything


def compose_image_transform(resolution: Tuple[int, int]) -> Compose:
    return Compose([
        lambda image: image.convert("RGB"),
        Resize(resolution),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class OutputBuffer:
    def __init__(self) -> None:
        self.outputs = []

    def __call__(self, module: Module, module_input: Tensor, module_output: Tensor) -> None:
        self.outputs.append(module_output[1].squeeze(0))

    def clear(self) -> None:
        self.outputs = []


class RSICDCaptionGenerator:
    def __init__(
            self,
            weights_filepath: Path,
            tokenizer_filepath: Path = TOKENIZER_FILE,
            config_filepath: Path = CONFIG_FILE,
    ) -> None:
        self.config = load_config(config_filepath)
        self.tokenizer = Tokenizer.init_from_file(tokenizer_filepath)
        self.device = get_available_device()
        seed_everything(self.config.seed)

        self.model = Image2Text(self.config, vocab_size=len(self.tokenizer.vocab)).to(self.device)
        self.model = load_weights(weights_filepath, self.model)
        self.model.eval()

        self.transform = compose_image_transform(self.config.image_size)
        self.transform_to_PIL = ToPILImage(mode="RGB")

        if self.model.decoder_type == "transformer":
            self.decoder_module = self.model.decoder.decoder
            self.output_buffers = [OutputBuffer() for _ in self.decoder_module.layers]

            for i, layer in enumerate(self.decoder_module.layers):
                attention_layer = getattr(layer, "multihead_attn")
                self._enable_transformer_attention_outputs(attention_layer)
                attention_layer.register_forward_hook(self.output_buffers[i])

    def caption_image_from_url(self, url: str, top_k: int = 1, show_attention: bool = False) -> str:
        return self.caption_image(Image.open(BytesIO(requests.get(url).content)), top_k, show_attention)

    def caption_image_from_file(self, image_filepath: Path, top_k: int = 1, show_attention: bool = False) -> str:
        return self.caption_image(Image.open(image_filepath), top_k, show_attention)

    def caption_image_from_tensor(self, tensor: Tensor, top_k: int = 1, show_attention: bool = False) -> str:
        return self.caption_image(self.transform_to_PIL(tensor), top_k, show_attention)

    def caption_image(self, image: Image, top_k: int = 1, show_attention: bool = False) -> str:
        if top_k < 1:
            raise ValueError("top_k must be greater than 0")

        with no_grad():
            encoder_output = self.model.encoder(self.transform(image).unsqueeze(0).to(self.device))

        if self.model.decoder_type == "transformer":
            token_ids = (
                self._greedy_inference_with_decoder_transformer(encoder_output)
                if top_k == 1
                else self._beam_search_inference_with_decoder_transformer(encoder_output, top_k)
            )
            caption = self.tokenizer.decode(token_ids)
            self._show_image(image, caption)

            if show_attention:
                self._visualize_transformer_attention(image, token_ids)

        elif self.model.decoder_type == "lstm":
            token_ids, alpha = (
                self._greedy_inference_with_decoder_lstm(encoder_output)
                if top_k == 1
                else self._beam_search_inference_with_decoder_lstm(encoder_output, top_k)
            )
            caption = self.tokenizer.decode(token_ids)
            self._show_image(image, caption)

            if show_attention:
                self._visualize_attention(image, token_ids, alpha)
        else:
            raise ValueError(f"Unknown decoder type: `{self.model.decoder_type}`")

        return caption

    def _greedy_inference_with_decoder_lstm(self, encoder_output: Tensor) -> Tuple[List[int], Tensor]:
        caption = [Vocab.SOS_ID, Vocab.EOS_ID]

        while True:
            caption_tensor = LongTensor(caption).unsqueeze(0).to(self.device)
            length_tensor = LongTensor([len(caption)]).unsqueeze(0).to(self.device)

            with no_grad():
                prediction, _, alpha, _ = self.model.decoder(encoder_output, caption_tensor, length_tensor)

            predicted_id = prediction.topk(1)[1].view(-1)[-1].item()

            if predicted_id == Vocab.EOS_ID:
                break

            caption.insert(-1, predicted_id)

        return caption, alpha

    def _greedy_inference_with_decoder_transformer(self, encoder_output: Tensor) -> List[int]:
        caption = [Vocab.SOS_ID]

        while True:
            caption_tensor = LongTensor(caption).unsqueeze(0).to(self.device)

            with no_grad():
                prediction = self.model.decoder(encoder_output, caption_tensor)

            predicted_id = prediction.topk(1)[1].view(-1)[-1].item()
            caption.append(predicted_id)

            if predicted_id == Vocab.EOS_ID:
                break

        return caption

    def _beam_search_inference_with_decoder_transformer(self, enc_output: Tensor, top_k: int) -> List[int]:
        beam = [(0, [Vocab.SOS_ID])]

        while True:
            candidates = []
            for score, sequence in beam:

                if sequence[-1] == Vocab.EOS_ID:
                    candidates.append((score, sequence))
                    continue

                caption_tensor = LongTensor(sequence).unsqueeze(0).to(self.device)

                with no_grad():
                    prediction = self.model.decoder(enc_output, caption_tensor)
                    log_probs = log_softmax(prediction[0, -1], dim=-1)

                top_k_log_probs, top_k_indices = log_probs.topk(top_k)

                for i in range(top_k):
                    next_score = score + top_k_log_probs[i].item()
                    next_seq = sequence + [top_k_indices[i].item()]
                    candidates.append((next_score, next_seq))

            beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:top_k]

            if all(seq[-1] == Vocab.EOS_ID for _, seq in beam):
                break

        return beam[0][1]

    def _beam_search_inference_with_decoder_lstm(self, enc_output: Tensor, top_k: int) -> Tuple[List[int], Tensor]:
        beam = [(0, [Vocab.SOS_ID, Vocab.EOS_ID], [])]

        while True:
            candidates = []
            for score, sequence, alphas in beam:

                if sequence[-2] == Vocab.EOS_ID:
                    candidates.append((score, sequence, alphas))
                    continue

                caption_tensor = LongTensor(sequence).unsqueeze(0).to(self.device)
                length_tensor = LongTensor([len(sequence)]).unsqueeze(0).to(self.device)

                with no_grad():
                    prediction, _, alpha, _ = self.model.decoder(enc_output, caption_tensor, length_tensor)
                    log_probs = log_softmax(prediction[0, -1], dim=-1)

                alphas.append(alpha)
                top_k_log_probs, top_k_indices = log_probs.topk(top_k)

                for i in range(top_k):
                    next_score = score + top_k_log_probs[i].item()
                    next_seq = sequence[:-1] + [top_k_indices[i].item()] + [Vocab.EOS_ID]
                    next_alphas = alphas + [alpha]
                    candidates.append((next_score, next_seq, next_alphas))

            beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:top_k]

            if all(seq[-2] == Vocab.EOS_ID for _, seq, _ in beam):
                break

        return beam[0][1], beam[0][2][-1]

    def _visualize_attention(self, image: Image, token_ids: List[int], alpha: Tensor) -> None:
        image = image.resize(self.config.image_size)
        tokens = [self.tokenizer.vocab.id_to_token.get(_id, Vocab.UNK_ID) for _id in token_ids]

        plt.figure(figsize=(16, 8))
        upscale_factor = self.config.image_size[0] // self.config.encoded_image_size[0]

        for token_idx in range(len(tokens)):
            if token_idx > 50:
                print("Too many generated tokens, stopping attention visualization.")
                break

            if 0 < token_idx < len(tokens) - 1:
                plt.subplot(int(np.ceil(len(tokens) / 5)), 5, token_idx)
                plt.text(0, 1, tokens[token_idx], color="black", backgroundcolor="white", fontsize=12)
                plt.set_cmap(plt.get_cmap("gray"))
                plt.axis("off")
                plt.imshow(image)

                token_alpha = alpha[0:, token_idx - 1, :].view(1, *self.config.encoded_image_size).squeeze(0)
                token_alpha = token_alpha.squeeze(0).cpu().numpy()
                token_alpha = skimage.transform.pyramid_expand(token_alpha, upscale=upscale_factor, sigma=12)
                # noinspection PyTypeChecker
                plt.imshow(token_alpha, alpha=0.8)

        plt.tight_layout()
        plt.show()

    def _visualize_transformer_attention(self, image: Image, token_ids: List[int]) -> None:
        alpha = [
            self.output_buffers[layer_idx].outputs[-1].mean(dim=0)
            for layer_idx in range(self.decoder_module.num_layers)
        ][-1].unsqueeze(0)

        self._visualize_attention(image, token_ids, alpha)

        for buffer in self.output_buffers:
            buffer.clear()

    @staticmethod
    def _show_image(image: Image, title: str) -> None:
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(title, fontsize=18)
        plt.show()

    @staticmethod
    def _enable_transformer_attention_outputs(module: Module) -> None:
        forward_method = module.forward

        def wrap(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False

            return forward_method(*args, **kwargs)

        module.forward = wrap
