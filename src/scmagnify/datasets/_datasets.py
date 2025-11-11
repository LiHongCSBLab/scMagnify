"""Datasets"""

from __future__ import annotations

import os
import shutil
import stat
from typing import TYPE_CHECKING

import pooch
from pooch import Decompress, Untar, Unzip

import scmagnify.logging as logg
from scmagnify.settings import settings

if TYPE_CHECKING:
    pass

__all__ = ["fetch_scm_data", "human_tcelldep_bm", "human_aki_ckd", "mouse_pancreas"]


def process_downloaded_file(fname, action, pooch):
    """
    Processes the downloaded file and returns a new file name.

    The function **must** take as arguments (in order):

    fname : str
        The full path of the file in the local data storage
    action : str
        Either: "download" (file doesn't exist and will be downloaded),
        "update" (file is outdated and will be downloaded), or "fetch"
        (file exists and is updated so no download is necessary).
    pooch : pooch.Pooch
        The instance of the Pooch class that is calling this function.

    The return value can be anything but is usually a full path to a file
    (or list of files). This is what will be returned by Pooch.fetch and
    pooch.retrieve in place of the original file path.
    """
    if action == "download":
        os.chmod(fname, stat.S_IRWXG | stat.S_IRWXU)
    if "tar" in fname:
        return Untar()(fname, action, pooch)
    elif "gz" in fname:
        return Decompress(method="gzip")(fname, action, pooch)
    elif "zip" in fname:
        return Unzip()(fname, action, pooch)
    return fname


# SCM_DATA = pooch.create(
#     path = pooch.os_cache("scmagnify"),
#     base_url = "",
#     env = "SCMAGNIFY_DATA",
#     registry = {"scmagnify_data.zip": "md5:a8d6c7712dd8b6fad7f6ce617d44c819",},
#     urls = {"scmagnify_data.zip": "https://zenodo.org/records/17440841/files/scmagnify_data.zip?download=1",}
# )


# def fetch_scm_data():
# """Fetch the scMagnify datasets.

# Returns
# -------
# str
#     The path to the scMagnify datasets.
# """
# data_path = SCM_DATA.fetch("scmagnify_data.zip", processor=process_downloaded_file)

# logg.info(f"scMagnify data are stored in `{data_path}`.")


def fetch_scm_data():
    """
    Fetch the scMagnify datasets, unzip, and organize them into the 'scm_data' directory.

    This function performs the following steps:
    1. Downloads 'scmagnify_data.zip' using pooch.
    2. Unzips it using the default pooch processor, which creates a
       'scmagnify_data.zip.unzip' directory.
    3. Moves the subdirectory 'scmagnify_data.zip.unzip/data/' to the cache root
       and renames it to 'scm_data/'.
    4. Cleans up the now-empty 'scmagnify_data.zip.unzip/' directory.

    Returns
    -------
    str
        The absolute path to the final 'scm_data' directory.
    """

    def unzip_and_move(fname, action, pooch_instance):
        """
        A custom processor for pooch.

        Parameters
        ----------
        fname : str
            The full path to the downloaded file (the .zip file).
        action : str
            The action being performed ("fetch").
        pooch_instance : pooch.Pooch
            The pooch instance that is calling the processor.

        Returns
        -------
        str
            The path to the final processed data directory.
        """
        cache_dir = pooch_instance.path
        final_data_path = os.path.join(cache_dir, "scm_data")

        if os.path.exists(final_data_path):
            logg.info(f"Data already processed and available at: {final_data_path}")
            return final_data_path

        unzipper = pooch.Unzip()
        unzipper(fname, action, pooch_instance)

        source_path = os.path.join(fname + ".unzip", "data")

        dest_path = final_data_path

        logg.info(f"Moving from '{source_path}' to '{dest_path}'...")

        if not os.path.isdir(source_path):
            raise FileNotFoundError(
                f"The expected 'data' directory was not found in the unzipped archive at '{source_path}'"
            )

        shutil.move(source_path, dest_path)

        try:
            shutil.rmtree(fname + ".unzip")
            logg.info(f"Cleaned up temporary directory: {fname + '.unzip'}")
            shutil.rmtree(fname)
            logg.info(f"Removed temporary zip file: {fname}")
        except OSError as e:
            logg.info(f"Could not clean up temporary directory {fname + '.unzip'}: {e}")

        return dest_path

    SCM_DATA = pooch.create(
        path=pooch.os_cache("scmagnify"),
        base_url="https://zenodo.org/records/17440841/files/",
        env="SCMAGNIFY_DATA",
        registry={
            "scmagnify_data.zip": "md5:a8d6c7712dd8b6fad7f6ce617d44c819",
        },
    )

    data_path = SCM_DATA.fetch("scmagnify_data.zip", processor=unzip_and_move)

    settings.scm_caches = os.environ.get("SCMAGNIFY_DATA", pooch.os_cache("scmagnify"))
    settings.scm_data = data_path

    return str(data_path)


# def scm_data():
#     """Get the path to the scMagnify datas.

#     Returns
#     -------
#     str
#         The path to the scMagnify datasets.
#     """
#     data_path = SCM_DATA.fetch("scmagnify_data.zip", processor=process_downloaded_file)
#     return str(data_path)


def human_tcelldep_bm():
    """Get the path to the human_tcelldep-bm dataset."""
    pass


def human_aki_ckd():
    """Get the path to the human_aki-ckd dataset."""
    pass


def mouse_pancreas():
    """Get the path to the mouse_pancreas dataset."""
    pass
