from argparse import Namespace

from torch import Tensor
from torch.nn import Module, Sequential, AdaptiveAvgPool2d
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    vgg16_bn,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    VGG16_BN_Weights,
)

ENCODER_BACKBONE = {
    "resnet18": {
        "model": resnet18,
        "weights": ResNet18_Weights.DEFAULT,
        "encoder_dim": 512,
    },
    "resnet34": {
        "model": resnet34,
        "weights": ResNet34_Weights.DEFAULT,
        "encoder_dim": 512,
    },
    "resnet50": {
        "model": resnet50,
        "weights": ResNet50_Weights.DEFAULT,
        "encoder_dim": 2048,
    },
    "resnet101": {
        "model": resnet101,
        "weights": ResNet101_Weights.DEFAULT,
        "encoder_dim": 2048,
    },
    "vgg16_bn": {
        "model": vgg16_bn,
        "weights": VGG16_BN_Weights.DEFAULT,
        "encoder_dim": 512,
    }
}


class Encoder(Module):
    def __init__(self, config: Namespace) -> None:
        super(Encoder, self).__init__()
        backbone = ENCODER_BACKBONE[config.encoder_backbone]
        self.encoder_dim = backbone["encoder_dim"]

        # Remove linear and pool layers (since we're not doing classification)
        self.resnet = Sequential(
            *list(backbone["model"](weights=backbone["weights"], progress=False).children())[:-2]
        )

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = AdaptiveAvgPool2d(config.encoded_image_size)

        for param in self.resnet.parameters():
            param.requires_grad = False

        if config.finetune_encoder:
            for layer in list(self.resnet.children())[5:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, images: Tensor) -> Tensor:
        # Input: (batch_size, 3, image_size, image_size)

        x = self.resnet(images)
        # (batch_size, 2048, image_size/32, image_size/32)

        x = self.adaptive_pool(x)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)

        x = x.permute(0, 2, 3, 1)
        # Output: (batch_size, encoded_image_size, encoded_image_size, encoder_dim)
        return x
