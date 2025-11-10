# settings/__init__.py

from scmagnify.settings._settings import settings
from scmagnify.settings._settings import set_workspace, set_genome, load_fonts
from scmagnify.settings._settings import autosave, autoshow
from scmagnify.settings._info import *

__all__ = ["settings", "autosave", "autoshow", "set_workspace", "set_genome", "info", "load_fonts"]