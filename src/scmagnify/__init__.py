from importlib.metadata import version

from . import datasets
from . import plotting as pl
from . import tools as tl
from .datasets import fetch_scm_data
from .GRNMuData import GRNMuData, read
from .models import (
    MAGNI,
    MSNGC,
    chromatin_constraint,
    get_edgedf,
    import_basalGRN,
    normalize_data,
    partial_ordering,
    to_nx,
    to_tensor,
)
from .settings import autosave, autoshow, info, load_fonts, set_genome, set_workspace, settings
from .tools._tensor_decomp import RegDecomp

__all__ = [
    "datasets",
    "pl",
    "tl",
    "GRNMuData",
    "read",
    "RegDecomp",
    "MAGNI",
    "MSNGC",
    "chromatin_constraint",
    "import_basalGRN",
    "partial_ordering",
    "normalize_data",
    "get_edgedf",
    "to_tensor",
    "to_nx",
    "fetch_scm_data",
    "settings",
    "autosave",
    "autoshow",
    "set_workspace",
    "set_genome",
    "info",
    "load_fonts",
    "__version__",
]

__version__ = version("scmagnify")
