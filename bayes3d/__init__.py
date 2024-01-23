"""
.. include:: ./documentation.md
"""
from .camera import *
from .likelihood import *
from .renderer import *
from .rgbd import *
from .transforms_3d import *
from .viz import *

try:
    import genjax

    from .genjax import *
except ImportError as e:
    print("GenJAX not installed. Importing bayes3d without genjax dependencies.")
    print(e)


RENDERER = None
