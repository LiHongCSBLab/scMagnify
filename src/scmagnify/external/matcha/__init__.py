"""MATCHA: Multi-omic Analysis of TF-CRE-gene Hierarchy for Activity.

This module integrates the MATCHA workflow for mapping gene programs to
co-accessible peaks and prioritizing transcription factors based on
chromatin accessibility and gene expression correlations.
"""

from __future__ import annotations

from ._matcha import MATCHA, GeneProgram, MatchaResult, PeakTFPrioritizer

__all__ = ["MATCHA", "GeneProgram", "MatchaResult", "PeakTFPrioritizer"]
