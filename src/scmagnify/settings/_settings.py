from __future__ import annotations

import logging
import os
import shutil
import sys
from enum import IntEnum
from pathlib import Path
from typing import TextIO

import matplotlib
import pooch
from matplotlib import font_manager as fm
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.theme import Theme
from rich.tree import Tree

from scmagnify.logging._logging import HINT, _LogFormatter, _RootLogger

__all__ = [
    "Verbosity",
    "scMagnifySettings",
    "settings",
    "autosave",
    "autoshow",
    "set_workspace",
    "set_genome",
    "load_fonts",
]

autosave = False
"""Save plots/figures as files in directory 'figs'.
Do not show plots/figures interactively.
"""

autoshow = True
"""Show all plots/figures automatically if autosave == False.
There is no need to call the matplotlib pl.show() in this case.
"""

# --------------
# Logging Setup
# --------------

custom_theme = Theme(
    {
        "logging.level.info": "bold green",  # INFO 级别颜色
        "logging.level.warning": "bold yellow",  # WARNING 级别颜色
        "logging.level.error": "bold red",  # ERROR 级别颜色
        "logging.level.debug": "bold orange_red1",  # DEBUG 级别颜色
    }
)


def _set_log_file(settings):
    file = settings.logfile
    name = settings.logpath
    root = settings._root_logger
    console = Console(theme=custom_theme)
    h = (
        RichHandler(
            markup=True,
            show_time=False,
            show_path=False,
            log_time_format="[%Y-%m-%d %H:%M:%S]",
            console=console,
        )
        if name is None
        else logging.FileHandler(name)
    )
    # h = logging.StreamHandler(file) if name is None else logging.FileHandler(name)
    h.setFormatter(_LogFormatter())
    h.setLevel(root.level)

    if len(root.handlers) == 1:
        root.removeHandler(root.handlers[0])
    elif len(root.handlers) > 1:
        raise RuntimeError("scMagnify's root logger somehow got more than one handler.")

    root.addHandler(h)


_VERBOSITY_LEVELS = {
    0: logging.ERROR,
    1: logging.WARNING,
    2: logging.INFO,
    3: HINT,
    4: logging.DEBUG,
}


class Verbosity(IntEnum):
    """Logging verbosity levels compatible with scMagnify's logger."""

    error = 0
    warning = 1
    info = 2
    hint = 3
    debug = 4

    @property
    def level(self) -> int:
        """Return the corresponding standard :mod:`logging` numeric level."""
        return _VERBOSITY_LEVELS[int(self)]


class scMagnifySettings:
    """Standalone configuration singleton for scMagnify (decoupled from Scanpy's private config)."""

    _instance: scMagnifySettings | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(
        self,
        verbosity: Verbosity | int | str = Verbosity.warning,
        file_format_figs: str = "pdf",
        logfile: TextIO | None = None,
        logpath: str | Path | None = None,
        frameon: bool = True,
        low_resolution_warning: bool = True,
    ) -> None:
        self._verbosity = self._parse_verbosity(verbosity)
        self.file_format_figs = file_format_figs
        self.logfile = sys.stderr if logfile is None else logfile
        self.logpath = Path(logpath) if logpath is not None else None
        self._frameon = frameon
        self._low_resolution_warning = low_resolution_warning
        self.n_jobs = 1

        # Workspace directories
        self.work_dir: str | None = None
        self.data_dir: str | None = None
        self.tmpfiles_dir: str | None = None
        self.log_dir: str | None = None
        self.genomes_dir: str | None = None
        self.models_dir: str | None = None
        self.figures_dir: str | None = None

        # Cached reference / tutorial data
        self.scm_caches: str | None = os.environ.get("SCMAGNIFY_DATA", None)
        self.scm_data: str | None = os.path.join(self.scm_caches, "scm_data") if self.scm_caches is not None else None

        # Reference genome files
        self.version: str | None = None
        self.gtf_file: str | None = None
        self.fasta_file: str | None = None
        self.tf_file: str | None = None

        self._root_logger = _RootLogger(self._verbosity.level)

    @staticmethod
    def _parse_verbosity(value: Verbosity | int | str) -> Verbosity:
        if isinstance(value, Verbosity):
            return value
        if isinstance(value, int):
            return Verbosity(value)
        if isinstance(value, str):
            key = value.lower()
            if key in Verbosity.__members__:
                return Verbosity[key]
        raise ValueError(f"Invalid verbosity {value!r}. Expected one of {list(Verbosity.__members__)} or 0..4.")

    @property
    def verbosity(self) -> Verbosity:
        """Current logging verbosity level."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value: Verbosity | int | str) -> None:
        parsed = self._parse_verbosity(value)
        self._verbosity = parsed
        self._root_logger.setLevel(parsed.level)
        for handler in self._root_logger.handlers:
            handler.setLevel(parsed.level)


settings = scMagnifySettings()
_set_log_file(settings)


def set_workspace(path, logging=False):
    """
    Set the working directory and create necessary subdirectories.

    Parameters
    ----------
    path (str): The path to the working directory.
    """
    # Ensure the path ends with a directory separator
    if not path.endswith(os.sep):
        path += os.sep

    # Create the main working directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    settings.work_dir = path
    # Define subdirectories
    data_dir = os.path.join(path, "data")
    tmpfiles_dir = os.path.join(path, "tmpfiles")
    models_dir = os.path.join(path, "models")
    figures_dir = os.path.join(path, "figures")

    # Create subdirectories if they don't exist
    for directory in [data_dir, tmpfiles_dir, models_dir, figures_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Update settings with the new directory paths
    settings.data_dir = data_dir
    settings.tmpfiles_dir = tmpfiles_dir
    settings.models_dir = models_dir
    settings.figures_dir = figures_dir

    # Update the log file path if necessary
    if logging:
        log_dir = os.path.join(path, "log")
        settings.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        settings.logpath = os.path.join(log_dir, "scmagnify.log")
        _set_log_file(settings)

    # Display the directory structure using rich.tree
    console = Console()
    tree = Tree(f"[bold white]workspace: {path}[/bold white]")
    data_node = tree.add("[white]data[/white]")
    models_node = tree.add("[white]models[/white]")
    tmpfiles_node = tree.add("[white]tmpfiles[/white]")
    figures_node = tree.add("[white]figures[/white]")

    if logging:
        log_node = tree.add("[white]log[/white]")
        log_node.add("[magenta]scmagnify.log[/magenta]")
    console.print(tree)


# -----------------------
# scMagnify Data Setup
# ------------------------
settings.scm_caches = os.environ.get("SCMAGNIFY_DATA", None)
settings.scm_data = os.path.join(settings.scm_caches, "scm_data") if settings.scm_caches is not None else None

# ----------------------
# Reference Genome Setup
# -----------------------
settings.version = None
settings.gtf_file = None
settings.fasta_file = None
settings.tf_file = None


def set_genome(version: str, provider: str = "UCSC", genomes_dir: str = None, download: bool = False):
    """
    Set the reference genome for the analysis using genomepy.

    Parameters
    ----------
    version : str
        The version of the reference genome to use.
    provider : str, optional
        The provider of the reference genome. Default is "UCSC".
    genomes_dir : str, optional
        The directory where the genome files are stored. Default is None.
    download : bool, optional
        If True, download the genome files if not found. Default is False.

    """
    import genomepy

    # Set the genome version in settings
    settings.version = version

    # Check if the genome is installed
    if genomes_dir is None:
        genomes_dir = settings.genomes_dir
        if genomes_dir is None:
            genomes_dir = os.path.join(settings.work_dir, "genomes")
            settings.genomes_dir = genomes_dir
    try:
        genomepy.Genome(version, genomes_dir=genomes_dir)
        # settings.gtf_file = os.path.join(genomes_dir, version, f"{version}.gtf")
        if settings.scm_data is None:
            raise FileNotFoundError(
                "scMagnify data directory not set. Please set SCMAGNIFY_DATA environment variable and run scm.datasets.fetch_scm_data()."
            )
        settings.gtf_file = os.path.join(settings.scm_data, "annotations", f"{version}.cellranger.gtf.gz")
        settings.fasta_file = os.path.join(genomes_dir, version, f"{version}.fa")
    except:
        if download:
            genomepy.install_genome(name=version, provider=provider, genomes_dir=genomes_dir)
            settings.gtf_file = os.path.join(genomes_dir, version, f"{version}.gtf")
            settings.fasta_file = os.path.join(genomes_dir, version, f"{version}.fa")
        else:
            raise FileNotFoundError(
                f"Genome files for {version} not found. \n Please download the genome files using genomepy.install_genome() or set download=True."
            )

    settings.tf_file = os.path.join(settings.scm_data, "tf_lists", f"allTFs_{version}.txt")
    if not os.path.exists(settings.tf_file):
        raise FileNotFoundError(
            f"Transcription factor list for {version} not found. \n Please download the TF list using scmagnify.download_tf_list() or set download=True."
        )

    console = Console()

    table = Table(title="Genome Information")
    table.add_column("Version", style="cyan")
    table.add_column("Provider", style="magenta")
    table.add_column("Directory", style="magenta")

    table.add_row(version, provider, genomes_dir)
    console.print(table)


def load_fonts(
    font_list: list[str], package_name: str = "scmagnify", font_dir: str = "fonts", clear_cache: bool = False
) -> list[str]:
    """
    Installs fonts from scMagnify's cache directory into matplotlib's font directory.

    This function finds .ttf files in the standard scMagnify cache directory and copies them
    into matplotlib's permanent font directory (`mpl-data/fonts/ttf`), making them
    available across all Python sessions. If fonts are newly installed, it will
    clear the font cache and recommend a kernel restart.

    Parameters
    ----------
    font_list : list[str]
        List of base font names to validate (e.g., ["Arial", "Helvetica"]).
    package_name : str, default="scmagnify"
        The cache namespace used to resolve the standard cache directory.
    font_dir : str, default="fonts"
        Directory within the scMagnify cache root containing .ttf font files.
    clear_cache : bool, default=False
        If True, forces a clear of the font cache even if no new fonts are installed.

    Returns
    -------
    list[str]
        Updated list of all available font names found in the system.
    """
    # --- Step 0: Locate Matplotlib's font directory ---
    mpl_font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    if not mpl_font_dir.exists():
        raise FileNotFoundError(f"Matplotlib font directory does not exist: {mpl_font_dir}")

    fonts_were_copied = False

    # --- Step 1: Locate the font directory within the standard cache root ---
    font_path = None
    try:
        cache_dir = Path(os.environ.get("SCMAGNIFY_DATA") or settings.scm_caches or pooch.os_cache(package_name))
        font_path = cache_dir / font_dir
        if not font_path.exists():
            print(f"⚠️ Warning: Cache font directory does not exist: {font_path}")
            font_path = None
    except Exception as e:
        print(f"⚠️ Warning: Could not access cache font directory: {e}")

    # --- Step 2: Search for and COPY fonts from the cache directory ---
    if font_path:
        # Check all base fonts requested by the user
        for base_font in font_list:
            # Find all font files matching the base name (e.g., Arial, Arial-Bold)
            font_files = list(font_path.glob(f"{base_font}*.ttf"))

            if not font_files:
                print(f"ℹ️ No font files matching '{base_font}*' found in cache font directory {font_path}")
                continue

            for font_file in font_files:
                dest_path = mpl_font_dir / font_file.name
                if dest_path.exists():
                    font_to_register = dest_path
                else:
                    try:
                        print(f"⚙️ Installing font '{font_file.name}'...")
                        shutil.copy2(font_file, dest_path)
                        print(f"✅ Copied '{font_file.name}' to matplotlib's font directory.")
                        fonts_were_copied = True
                        font_to_register = dest_path
                    except Exception as e:
                        print(f"⚠️ Failed to copy font from {font_file}: {e}")
                        font_to_register = font_file

                try:
                    fm.fontManager.addfont(str(font_to_register))
                except Exception as e:
                    print(f"⚠️ Failed to register font {font_to_register}: {e}")

    # --- Step 3: Clear cache if new fonts were installed or if requested ---
    if fonts_were_copied or clear_cache:
        if fonts_were_copied:
            print("\n🔥 New fonts were installed. Clearing matplotlib font cache to apply changes.")
        else:
            print("🗑️ Clearing matplotlib font cache as requested...")

        try:
            cache_dir = matplotlib.get_cachedir()
            for file in os.listdir(cache_dir):
                if file.startswith("fontlist-"):
                    file_path = os.path.join(cache_dir, file)
                    os.remove(file_path)
                    print(f"   - Removed cache file: {file}")
            print("✅ Font cache cleared.")
            print("👉 IMPORTANT: Please RESTART your Python kernel (e.g., in Jupyter) for changes to take full effect.")
        except Exception as e:
            print(f"⚠️ Could not clear font cache automatically: {e}")
            print(f"   Consider manually deleting files from: {matplotlib.get_cachedir()}")

    # --- Step 4: Return the list of available fonts ---
    final_available_fonts = {f.name for f in fm.fontManager.ttflist}
    valid_fonts = set()
    for base_font in font_list:
        for f_name in final_available_fonts:
            if f_name.startswith(base_font):
                valid_fonts.add(f_name)

    if not valid_fonts:
        valid_fonts.add("sans-serif")
        print("No valid fonts found; falling back to 'sans-serif'")

    return sorted(valid_fonts)
