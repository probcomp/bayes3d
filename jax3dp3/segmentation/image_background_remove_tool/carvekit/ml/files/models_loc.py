"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import pathlib
from ...ml.files import checkpoints_dir
from ...utils.download_models import downloader


def u2net_full_pretrained() -> pathlib.Path:
    """Returns u2net pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("u2net.pth")


def basnet_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("basnet.pth")


def deeplab_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("deeplab.pth")


def fba_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("fba_matting.pth")


def tracer_b7_pretrained() -> pathlib.Path:
    """Returns TRACER with EfficientNet v1 b7 encoder pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("tracer_b7.pth")


def tracer_hair_pretrained() -> pathlib.Path:
    """Returns TRACER with EfficientNet v1 b7 encoder model for hair segmentation location

    Returns:
        pathlib.Path to model location
    """
    return downloader("tracer_hair.pth")


def download_all():
    u2net_full_pretrained()
    fba_pretrained()
    deeplab_pretrained()
    basnet_pretrained()
    tracer_b7_pretrained()
