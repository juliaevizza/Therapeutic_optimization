from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PredictorConfig, ProjectPaths
from ..io import read_single_fasta
from .base import UbiquitinationPredictor
from .eup import EUPPredictor


def build_predictor(config: PredictorConfig) -> UbiquitinationPredictor:
    
def run_up1(
    predictor: UbiquitinationPredictor,
    paths: ProjectPaths,
) -> pd.DataFrame:
    """UP1: score the WT sequence."""
    protein_id, sequence = read_single_fasta(paths.wt_fasta)
    result = predictor.predict_sequence(
        sequence=sequence,
        protein_id=protein_id,
        variant_id='WT',
        output_dir=paths.ubi_wt,
    )
    output = paths.table('UP1_wt_ubiquitination.csv')
    result.to_csv(output, index=False)
    return result


#TODO finish threading into this module, moved from lysine free generation 
def _prediction_summary(predictions: pd.DataFrame, ubi_threshold: float) -> dict[str, float | int]:
    "Returns data from ubiquitnatin prediciton "

    if predictions.empty:
        return {
            'lysine_count': 0,
            'positive_site_count': 0,
        }
    probabilities = predictions['probability'].astype(float)
    positive = probabilities > ubi_threshold
    return {
        'lysine_count': int(len(predictions)),
        'positive_site_count': int(len(positive)),
        'residues of interest': positive
    }