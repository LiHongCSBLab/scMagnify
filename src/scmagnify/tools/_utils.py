from __future__ import annotations

from typing import TYPE_CHECKING

## import other packages
import numpy as np
import pandas as pd
import networkx as nx

## from scmagnify import ..
from scmagnify import logging as logg

if TYPE_CHECKING:
    from typing import Literal, Union, Optional, List
    from anndata import AnnData
    from mudata import MuData

__all__ = ["get_network"]




