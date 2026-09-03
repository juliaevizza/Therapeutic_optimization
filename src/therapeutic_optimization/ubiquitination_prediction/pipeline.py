from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PredictorConfig, ProjectPaths
from ..io import read_single_fasta
from .base import UbiquitinationPredictor
from .eup import EUPPredictor


def build_predictor(config: PredictorConfig) -> UbiquitinationPredictor:
    name = config.name.lower()
    if name == 'eup':
        return EUPPredictor(
            threshold=config.threshold,
            eup_repo_dir=config.eup_repo_dir,
            model_cache_dir=config.model_cache_dir,
            force_clone=config.force_clone_eup,
        )
    raise ValueError(
        f'Unknown ubiquitination predictor {config.name!r}. '
        'Add an adapter implementing UbiquitinationPredictor and register it in build_predictor().'
    )


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
    output = paths.tables / 'UP1_wt_ubiquitination.csv'
    result.to_csv(output, index=False)
    return result


def run_ub2(
    predictor: UbiquitinationPredictor,
    s1_conserved: pd.DataFrame,
    paths: ProjectPaths,
) -> pd.DataFrame:
    """UB2: re-score only mutants that passed S1 structural preservation."""
    required = {'variant_id', 'fasta_path'}
    missing = required - set(s1_conserved.columns)
    if missing:
        raise ValueError(f'S1 is missing required columns: {sorted(missing)}')

    frames: list[pd.DataFrame] = []
    for row in s1_conserved.itertuples(index=False):
        variant_id = str(row.variant_id)
        protein_id, sequence = read_single_fasta(Path(row.fasta_path))
        variant_dir = paths.ubi_mutants / variant_id
        frame = predictor.predict_sequence(
            sequence=sequence,
            protein_id=protein_id,
            variant_id=variant_id,
            output_dir=variant_dir,
        )
        frames.append(frame)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        from .base import STANDARD_COLUMNS
        combined = pd.DataFrame(columns=STANDARD_COLUMNS)

    combined.to_csv(paths.tables / 'UB2_mutant_ubiquitination.csv', index=False)
    return combined
