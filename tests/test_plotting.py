"""Smoke tests for scmagnify.plotting functions using non-interactive Agg backend."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scmagnify as scm


def test_barplot_smoke(small_gdata):
    scm.tl.get_network_score(small_gdata, modal="GRN", key_added="network_score", source_only=True)
    scm.pl.barplot(small_gdata, modal="GRN", key="network_score", n_top=3, ncols=2, show=False)
    plt.close("all")


def test_rankplot_smoke(small_gdata):
    scm.tl.get_network_score(small_gdata, modal="GRN", key_added="network_score", source_only=True)
    scm.pl.rankplot(small_gdata, modal="GRN", key="network_score", n_top=2, ncols=2, show=False)
    plt.close("all")


def test_distplot_smoke():
    rng = np.random.default_rng(0)
    data_dict = {
        "ScoreA": pd.Series(rng.normal(size=50)),
        "ScoreB": pd.Series(rng.uniform(0, 1, size=50)),
    }
    thresholds = {"ScoreA": 0.5, "ScoreB": 0.7}
    scm.pl.distplot(data_dict=data_dict, thresholds=thresholds, ncols=2, show=False)
    plt.close("all")
