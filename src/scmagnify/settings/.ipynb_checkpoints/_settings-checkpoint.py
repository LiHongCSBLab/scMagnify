import os
import copy
import logging
from rich.logging import RichHandler
from rich.table import Table
from rich.console import Console
from rich.tree import Tree
from rich.theme import Theme

from scanpy import _settings as scanpy_settings
import scmagnify as scm
from scmagnify.logging._logging import _LogFormatter, _RootLogger

__all__ = ["settings", "autosave", "autoshow", "set_workspace", "figdir", "set_genome"]

autosave = False
"""Save plots/figures as files in directory 'figs'.
Do not show plots/figures interactively.
"""

autoshow = True
"""Show all plots/figures automatically if autosave == False.
There is no need to call the matplotlib pl.show() in this case.
"""

figdir = "./"

# --------------
# Logging Setup
# --------------

custom_theme = Theme({
    "logging.level.info": "bold green",  # INFO 级别颜色
    "logging.level.warning": "bold yellow",  # WARNING 级别颜色
    "logging.level.error": "bold red",  # ERROR 级别颜色
    "logging.level.debug": "bold orange_red1",  # DEBUG 级别颜色
})

def _set_log_file(settings):
    file = settings.logfile
    name = settings.logpath
    root = settings._root_logger
    console = Console(theme=custom_theme)
    h = RichHandler(markup=True,
                    show_time=False,
                    show_path=False,
                    log_time_format="[%Y-%m-%d %H:%M:%S]",
                    console=console,
                    ) if name is None else logging.FileHandler(name)
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

class scMagnifySettings(scanpy_settings.ScanpyConfig):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(scMagnifySettings, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, *args, **kwargs):
        super(scMagnifySettings, self).__init__(*args, **kwargs)
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

def set_workspace(path, logging=False):
    """
    Set the working directory and create necessary subdirectories.

    Parameters:
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
    data_dir = os.path.join(path, 'data')
    tmpfiles_dir = os.path.join(path, 'tmpfiles')
    models_dir = os.path.join(path, 'models')

    # Create subdirectories if they don't exist
    for directory in [data_dir, tmpfiles_dir, models_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Update settings with the new directory paths
    settings.data_dir = data_dir
    settings.tmpfiles_dir = tmpfiles_dir
    settings.models_dir = models_dir

    # Update the log file path if necessary
    if logging:
        log_dir = os.path.join(path, 'log')
        settings.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        settings.logpath = os.path.join(log_dir, 'scmagnify.log')
        _set_log_file(settings)

    # Display the directory structure using rich.tree
    console = Console()
    tree = Tree(f"[bold white]workspace: {path}[/bold white]")
    data_node = tree.add(f"[white]data[/white]")
    models_node = tree.add(f"[white]models[/white]")
    tmpfiles_node = tree.add(f"[white]tmpfiles[/white]")

    if logging:
        log_node = tree.add(f"[white]log[/white]")
        log_node.add(f"[magenta]scmagnify.log[/magenta]")
    console.print(tree)


# ----------------------
# Reference Genome Setup
# -----------------------
settings.version = None
settings.gtf_file = None
settings.fasta_file = None
settings.tf_file = None

def set_genome(version: str,
               provider: str = "UCSC",
               genomes_dir: str = None,
               download: bool = False):
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
        settings.gtf_file = os.path.join(genomes_dir, version, f"{version}.gtf")
        settings.fasta_file = os.path.join(genomes_dir, version, f"{version}.fa")
    except:
        if download:
            genomepy.install_genome(name=version, provider=provider, genomes_dir=genomes_dir)
            settings.gtf_file = os.path.join(genomes_dir, version, f"{version}.gtf")
            settings.fasta_file = os.path.join(genomes_dir, version, f"{version}.fa")
        else:
            raise FileNotFoundError(f"Genome files for {version} not found. \n Please download the genome files using genomepy.install_genome() or set download=True.")

    scm_dir = os.path.dirname(scm.__file__)
    settings.tf_file = os.path.join(scm_dir, "data", "tf_lists", f"allTFs_{version}.txt")
    if not os.path.exists(settings.tf_file):
        raise FileNotFoundError(f"Transcription factor list for {version} not found. \n Please download the TF list using scmagnify.download_tf_list() or set download=True.")

    console = Console()

    table = Table(title="Genome Information")
    table.add_column("Version", style="cyan")
    table.add_column("Provider", style="magenta")
    table.add_column("Directory", style="magenta")

    table.add_row(version, provider, genomes_dir)
    console.print(table)
