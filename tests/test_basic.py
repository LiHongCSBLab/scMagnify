"""Basic package import and metadata tests."""

import pytest

import scmagnify as scm


def test_package_has_version():
    assert isinstance(scm.__version__, str)
    assert len(scm.__version__) > 0


def test_public_namespace_exports():
    for name in ("tl", "pl", "datasets", "GRNMuData", "read", "RegDecomp", "MAGNI", "MSNGC", "settings"):
        assert hasattr(scm, name)


@pytest.mark.parametrize("loader", ["human_tcelldep_bm", "human_aki_ckd", "mouse_pancreas"])
def test_unimplemented_dataset_loaders_raise(loader):
    with pytest.raises(NotImplementedError, match="fetch_scm_data"):
        getattr(scm.datasets, loader)()
