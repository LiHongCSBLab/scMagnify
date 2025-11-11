import base64
import re
import sys
import urllib.request
from collections import OrderedDict
from importlib.resources import files

from scmagnify.settings import settings

__all__ = ["info"]


def supports_html() -> bool:
    """Test whether current runtime supports HTML output (e.g., Jupyter notebook)."""
    if "IPython" not in sys.modules or "IPython.display" not in sys.modules:
        return False
    from IPython.display import display

    class DisplayInspector:
        def __init__(self) -> None:
            self.status = None

        def _repr_html_(self) -> str:
            self.status = "HTML"
            return ""

        def __repr__(self) -> str:
            self.status = "Plain"
            return ""

    inspector = DisplayInspector()
    display(inspector)
    return inspector.status == "HTML"


def get_latest_version(
    url="https://raw.githubusercontent.com/your-username/your-repo/main/pyproject.toml",
    timeout=5,
) -> str | None:
    """Fetch the latest package version from a remote pyproject.toml file."""
    try:
        response = urllib.request.urlopen(url, timeout=timeout)
        content = response.read().decode()
        match = re.search(r'version\s*=\s*"(.*)"', content)
        if match:
            return match.group(1)
        return None
    except Exception:
        return None


def _get_info() -> dict[str, dict]:
    """Collect package information, including version and dependencies."""
    from scmagnify import __version__  # Package name

    info: dict[str, dict] = OrderedDict()

    # Version information
    latest = get_latest_version()
    if latest == __version__:
        version_message = f"v{__version__} (up to date)"
    else:
        version_message = f"v{__version__} (latest: v{latest})" if latest else f"v{__version__}"
    info["package_version"] = {
        "title": "Installed version",
        "message": version_message,
        "value": __version__,
    }

    # Key dependencies: scanpy, mudata, cellrank
    info["dependencies"] = {
        "title": "Key dependencies",
        "message": "",
        "value": [],
    }
    try:
        import scanpy

        info["dependencies"]["message"] += f"scanpy v{scanpy.__version__}, "
        info["dependencies"]["value"].append(("scanpy", scanpy.__version__))
    except ImportError:
        pass
    try:
        import mudata

        info["dependencies"]["message"] += f"mudata v{mudata.__version__}, "
        info["dependencies"]["value"].append(("mudata", mudata.__version__))
    except ImportError:
        pass
    try:
        import cellrank

        info["dependencies"]["message"] += f"cellrank v{cellrank.__version__}, "
        info["dependencies"]["value"].append(("cellrank", cellrank.__version__))
    except ImportError:
        pass
    try:
        import decoupler

        info["dependencies"]["message"] += f"decoupler v{decoupler.__version__}, "
        info["dependencies"]["value"].append(("decoupler", decoupler.__version__))
    except ImportError:
        pass
    try:
        import SEACells

        info["dependencies"]["message"] += f"SEACells v{SEACells.__version__}, "
        info["dependencies"]["value"].append(("SEACells", SEACells.__version__))
    except ImportError:
        pass

    info["dependencies"]["message"] = info["dependencies"]["message"].rstrip(", ") or "No key dependencies detected"

    # Optional dependency: PyTorch
    info["pytorch_version"] = {
        "title": "PyTorch version",
        "message": "PyTorch not installed. Install with 'pip install torch' or 'conda install pytorch'.",
        "value": None,
    }

    try:
        import torch

        info["pytorch_version"]["message"] = f"v{torch.__version__}"
        info["pytorch_version"]["value"] = torch.__version__
    except ImportError:
        pass

    info["cuda_available"] = {
        "title": "CUDA available",
        "message": "PyTorch not installed or CUDA not available.",
        "value": False,
    }

    try:
        if torch.cuda.is_available():
            info["cuda_available"]["message"] = True
            info["cuda_available"]["value"] = True
        else:
            info["cuda_available"]["message"] = False
            info["cuda_available"]["value"] = False
    except NameError:
        # PyTorch not installed
        pass

    info["scm_data_cached"] = {
        "title": "scmagnify data cached",
        "message": False,
        "value": False,
    }

    try:
        if settings.scm_data is not None:
            info["scm_data_cached"]["message"] = True
            info["scm_data_cached"]["value"] = True
    except AttributeError:
        pass

    # Repository URL
    info["repo_url"] = {
        "title": "Repository",
        "message": "https://github.com/your-username/your-repo",
        "value": "https://github.com/your-username/your-repo",
    }

    return info


def info():
    """Display package information, with HTML output in Jupyter or plain text in terminal.

    References
    ----------
    [1] The `corneto` library, specifically the `info` utility function.
        Source: https://github.com/saezlab/corneto/blob/bb12cc977896be2d3a6b0d196605ebaa7b0f2d46/corneto/_util.py

    """
    info = _get_info()

    if supports_html():
        from IPython.display import HTML, display

        # Load logo
        logo_path = files("scmagnify").joinpath("data/logo_min.png")
        try:
            with open(logo_path, "rb") as f:
                img_bytes = f.read()
            b64img = base64.b64encode(img_bytes).decode("utf-8")
            logo_html = f'<img src="data:image/png;base64,{b64img}" style="width: 100%; max-width:100px;" />'
        except FileNotFoundError:
            logo_html = "Logo not found"

        # Build HTML table
        html_info = ""
        for k, v in info.items():
            title = v["title"]
            message = v["message"]
            if "_url" in k:
                message = f"<a href={message}>{message}</a>"
            html_info += f"<tr><td>{title}:</td><td style='text-align:left'>{message}</td></tr>"
        html = f"""
        <table style='background-color:rgba(0, 0, 0, 0);'>
        <tr>
            <td style="min-width:85px">{logo_html}</td>
            <td>
            <table>{html_info}</table>
            </td>
        </tr>
        </table>"""
        display(HTML(html))
    else:
        for v in info.values():
            print(f"{v['title']}: {v['message']}")
