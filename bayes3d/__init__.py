"""
.. include:: ./documentation.md
"""
from .transforms_3d import *
from .renderer import *
from .rgbd import *
from .likelihood import *
from .camera import *
from .viz import *
from . import utils
from . import distributions
from . import scene_graph
from . import colmap

try:
    import genjax
    from .genjax import *
except ImportError:
    print("GenJAX not installed. Importing bayes3d without genjax dependencies.")


RENDERER = None