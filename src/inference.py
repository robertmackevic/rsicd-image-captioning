from io import BytesIO
from pathlib import Path
from typing import Tuple, List

import numpy as np
import requests
import skimage
from PIL import Image
from matplotlib import pyplot as plt
from torch import LongTensor, no_grad, Tensor
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

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

    def caption_image_from_url(self, url: str, show_attention: bool = False) -> str:
        return self.caption_image(Image.open(BytesIO(requests.get(url).content)), show_attention)

    def caption_image_from_file(self, image_filepath: Path, show_attention: bool = False) -> str:
        return self.caption_image(Image.open(image_filepath), show_attention)

    def caption_image(self, image: Image, show_attention: bool = False) -> str:
        with no_grad():
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            encoder_output = self.model.encoder(image_tensor)
            caption = [Vocab.SOS_ID, Vocab.EOS_ID]

            while True:
                caption_tensor = LongTensor(caption).unsqueeze(0).to(self.device)
                length_tensor = LongTensor([len(caption)]).unsqueeze(0).to(self.device)

                prediction, _, _, alpha, _ = self.model.decoder(encoder_output, caption_tensor, length_tensor)
                predicted_id = prediction.topk(1)[1].view(-1)[-1].item()

                if predicted_id == Vocab.EOS_ID:
                    break

                caption.insert(-1, predicted_id)

        decoded_caption = self.tokenizer.decode(caption)

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(decoded_caption)
        plt.show()

        if show_attention:
            self._visualize_attention(image, caption, alpha)

        return decoded_caption

    def _visualize_attention(self, image: Image, predicted_ids: List[int], alpha: Tensor) -> None:
        tokens = [self.tokenizer.vocab.id_to_token.get(_id, Vocab.UNK_ID) for _id in predicted_ids]

        plt.figure(figsize=(16, 8))
        upscale_factor = self.config.image_size[0] // self.config.encoded_image_size[0]

        for token_idx in range(len(tokens)):
            if token_idx > 50:
                print("Too many generated tokens, stopping attention visualization.")
                break

            plt.subplot(int(np.ceil(len(tokens) / 5)), 5, token_idx + 1)
            plt.text(0, 1, tokens[token_idx], color="black", backgroundcolor="white", fontsize=12)
            plt.set_cmap(plt.get_cmap("gray"))
            plt.axis("off")
            plt.imshow(image)

            if token_idx != 0:
                token_alpha = alpha[0:, token_idx - 1, :].view(1, *self.config.encoded_image_size).squeeze(0)
                token_alpha = token_alpha.squeeze(0).cpu().numpy()
                token_alpha = skimage.transform.pyramid_expand(token_alpha, upscale=upscale_factor, sigma=12)
                # noinspection PyTypeChecker
                plt.imshow(token_alpha, alpha=0.8)

        plt.show()
