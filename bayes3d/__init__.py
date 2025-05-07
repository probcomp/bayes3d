"""
.. include:: ./documentation.md
"""

from importlib import metadata

from . import colmap, distributions, scene_graph, utils
from .camera import *
from .likelihood import *
from .renderer import *
from .rgbd import *
from .transforms_3d import *
from .viz import *

RENDERER: "Renderer" = None

__version__ = metadata.version("bayes3d")

__all__ = ["colmap", "distributions", "scene_graph", "utils"]
