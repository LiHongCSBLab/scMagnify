"""Unit tests for standalone scMagnifySettings (decoupled from ScanpyConfig)."""

from __future__ import annotations

import logging

import pytest

from scmagnify.settings import set_workspace, settings
from scmagnify.settings._settings import Verbosity, scMagnifySettings


def test_settings_is_standalone_singleton():
    # Must not inherit from scanpy's private ScanpyConfig
    base_names = [cls.__name__ for cls in scMagnifySettings.__mro__]
    assert "ScanpyConfig" not in base_names
    assert scMagnifySettings() is settings


def test_verbosity_levels_and_setter():
    original = settings.verbosity
    try:
        settings.verbosity = "debug"
        assert settings.verbosity is Verbosity.debug
        assert settings.verbosity.level == logging.DEBUG
        assert settings._root_logger.level == logging.DEBUG

        settings.verbosity = 0
        assert settings.verbosity is Verbosity.error
        assert settings.verbosity.level == logging.ERROR

        settings.verbosity = Verbosity.info
        assert settings.verbosity.level == logging.INFO

        with pytest.raises(ValueError, match="Invalid verbosity"):
            settings.verbosity = "not_a_level"
    finally:
        settings.verbosity = original


def test_set_workspace_creates_directories_and_logfile(tmp_path):
    old_work_dir = settings.work_dir
    old_logpath = settings.logpath
    try:
        ws = tmp_path / "scm_ws"
        set_workspace(str(ws), logging=True)
        assert (ws / "data").is_dir()
        assert (ws / "tmpfiles").is_dir()
        assert (ws / "models").is_dir()
        assert (ws / "figures").is_dir()
        assert (ws / "log").is_dir()
        assert settings.logpath == str(ws / "log" / "scmagnify.log")
    finally:
        settings.work_dir = old_work_dir
        settings.logpath = old_logpath
