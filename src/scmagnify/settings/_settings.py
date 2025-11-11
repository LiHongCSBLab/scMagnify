import logging
import os
import shutil
from pathlib import Path

import matplotlib
from matplotlib import font_manager as fm
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.theme import Theme
from rich.tree import Tree
from scanpy import _settings as scanpy_settings

from scmagnify.logging._logging import _LogFormatter, _RootLogger

__all__ = ["settings", "autosave", "autoshow", "set_workspace", "set_genome", "load_fonts"]

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


# settings = copy.copy(settings)
# settings._root_logger = _RootLogger(settings.verbosity)
# # these 2 lines are necessary to get it working (otherwise no logger is found)
# # this is a hacky way of modifying the logging, in the future, use our own
# _set_log_file(settings)

# settings.verbosity = settings.verbosity

# TODO update scanpy and use the new settings class
# class scMagnifySettings:
#     def __init__(self):
#         # Scanpy settings作为属性暴露
#         self._scanpy_settings = sc.settings

#         # 你的扩展属性
#         self.data_dir = None
#         self.tmpfiles_dir = None
#         self.log_dir = None
#         self.genomes_dir = None
#         self.models_dir = None
#         self.figures_dir = None
#         self.work_dir = None
#         self.version = None
#         self.gtf_file = None
#         self.fasta_file = None
#         self.tf_file = None

#         # 其他自定义属性/方法
#         # ...

#     # 代理 scanpy settings 的属性
#     def __getattr__(self, attr):
#         # 如果属性属于 scanpy settings，则转发
#         return getattr(self._scanpy_settings, attr)

#     def __setattr__(self, attr, value):
#         # 允许设置本地属性和 scanpy settings 属性
#         # 注意，__setattr__在__init__期间会被调用，所以要特殊处理
#         if attr in [
#             "data_dir", "tmpfiles_dir", "log_dir", "genomes_dir",
#             "models_dir", "figures_dir", "work_dir", "version",
#             "gtf_file", "fasta_file", "tf_file", "_scanpy_settings"
#         ]:
#             object.__setattr__(self, attr, value)
#         else:
#             # 其他属性转发到 scanpy settings
#             setattr(self._scanpy_settings, attr, value)

#     # 你可以继续写自己的方法，如 set_workspace、set_genome、load_fonts 等，
#     # 这些方法就直接用 self 或 self._scanpy_settings 访问属性即可

# settings = scMagnifySettings()


class scMagnifySettings(scanpy_settings.ScanpyConfig):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._root_logger = _RootLogger(self.verbosity)


settings = scMagnifySettings()
_set_log_file(settings)
# ---------------------
# Work Directory Setup
# ---------------------
settings.data_dir = None
settings.tmpfiles_dir = None
settings.log_dir = None
settings.genomes_dir = None
settings.models_dir = None
settings.figures_dir = None


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
    Installs fonts from a package into matplotlib's font directory.

    This function finds .ttf files in a specified package directory and copies them
    into matplotlib's permanent font directory (`mpl-data/fonts/ttf`), making them
    available across all Python sessions. If fonts are newly installed, it will
    clear the font cache and recommend a kernel restart.

    Parameters
    ----------
    font_list : list[str]
        List of base font names to validate (e.g., ["Arial", "Helvetica"]).
    package_name : str, default="scmagnify"
        The name of your Python package where the fonts are located.
    font_dir : str, default="data/fonts"
        Directory within the package containing .ttf font files.
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

    # --- Step 1: Locate the font directory within the installed package ---
    font_path = None
    try:
        # from importlib.resources import files
        # font_path = files(package_name).joinpath(font_dir)
        font_path = Path(os.path.join(settings.scm_data, font_dir))
        if not font_path.exists():
            print(f"⚠️ Warning: Package font directory does not exist: {font_path}")
            font_path = None
    except Exception as e:
        print(f"⚠️ Warning: Could not access package font directory: {e}")

    # --- Step 2: Search for and COPY fonts from the package directory ---
    if font_path:
        # Check all base fonts requested by the user
        for base_font in font_list:
            # Find all font files matching the base name (e.g., Arial, Arial-Bold)
            font_files = list(font_path.glob(f"{base_font}*.ttf"))

            if not font_files:
                print(f"ℹ️ No font files matching '{base_font}*' found in package directory {font_path}")
                continue

            for font_file in font_files:
                dest_path = mpl_font_dir / font_file.name
                if dest_path.exists():
                    # Font is already installed, skip.
                    continue
                try:
                    # --- This is the new logic: Copy the font file ---
                    print(f"⚙️ Installing font '{font_file.name}'...")
                    shutil.copy2(font_file, dest_path)
                    print(f"✅ Copied '{font_file.name}' to matplotlib's font directory.")
                    fonts_were_copied = True
                except Exception as e:
                    print(f"⚠️ Failed to copy font from {font_file}: {e}")

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
