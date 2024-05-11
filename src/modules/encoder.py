from torch import Tensor
from torch.nn import Module, Sequential, AdaptiveAvgPool2d
from torchvision.models import resnet101

from src.utils import load_config


class Encoder(Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        config = load_config()

        # Remove linear and pool layers (since we're not doing classification)
        self.resnet = Sequential(*list(resnet101(pretrained=True).children())[:-2])

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = AdaptiveAvgPool2d(config.encoded_image_size)

        for param in self.resnet.parameters():
            param.requires_grad = False

        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        if config.finetune_encoder:
            for layer in list(self.resnet.children())[5:]:
                for param, in layer.parameters():
                    param.requires_grad = True

    def forward(self, images: Tensor) -> Tensor:
        # Input: (batch_size, 3, image_size, image_size)

        x = self.resnet(images)
        # (batch_size, 2048, image_size/32, image_size/32)

        x = self.adaptive_pool(x)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)

        x = x.permute(0, 2, 3, 1)
        # Output: (batch_size, encoded_image_size, encoded_image_size, 2048)
        return x
