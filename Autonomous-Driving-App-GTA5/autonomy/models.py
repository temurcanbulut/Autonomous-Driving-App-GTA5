"""Explicit supported perception architectures; runtime never downloads weights."""

ARCHITECTURES = ("lraspp_mobilenet_v3_large", "deeplabv3_mobilenet_v3_large")


def build_model(name, classes=19):
    from torchvision.models.segmentation import (
        deeplabv3_mobilenet_v3_large,
        lraspp_mobilenet_v3_large,
    )

    if name == ARCHITECTURES[0]:
        return lraspp_mobilenet_v3_large(num_classes=classes, weights=None, weights_backbone=None)
    if name == ARCHITECTURES[1]:
        return deeplabv3_mobilenet_v3_large(
            num_classes=classes, weights=None, weights_backbone=None, aux_loss=False
        )
    raise ValueError("Unsupported perception architecture")
