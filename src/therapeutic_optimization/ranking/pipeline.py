from __future__ import annotations

import math

import pandas as pd

from ..config import ESM2AnalysisConfig, ProjectPaths
from ..io import parse_mutation


def run_r1(
    t2_manifest: pd.DataFrame,
    s1_metrics: pd.DataFrame,
    paths: ProjectPaths,
) -> pd.DataFrame:
    r1 = t2_manifest.merge(structural, on='variant_id', how='left')

