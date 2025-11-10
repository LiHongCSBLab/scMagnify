from importlib.metadata import version

__all__ = ["pl", "tl", "GRNMuData"]

from .models import *
from . import plotting as pl
from . import tools as tl

from scmagnify.tools._tensor_decomp import RegDecomp
from scmagnify.GRNMuData import GRNMuData, read
from scmagnify.settings import *
from scmagnify.utils import *
from scmagnify.datasets import *

# __version__ = version("scMagnify")
__version__ = "0.0.0"