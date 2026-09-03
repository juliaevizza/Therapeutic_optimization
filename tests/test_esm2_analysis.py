from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from therapeutic_optimization.config import ESM2AnalysisConfig, ProjectPaths
from therapeutic_optimization.esm2_analysis import SequenceScore, run_esm2_analysis
from therapeutic_optimization.io import write_fasta


class FakeESM2Scorer:
    def __init__(self, save_per_residue: bool = True) -> None:
        self.config = ESM2AnalysisConfig(
            model_name='fake/esm2',
            save_per_residue=save_per_residue,
        )

    def score_sequence(self, sequence: str) -> SequenceScore:
        codes = np.asarray([ord(amino_acid) for amino_acid in sequence], dtype=float)
        log_probabilities = -codes / 100.0
        representations = np.column_stack([codes, codes ** 2 / 100.0])
        return SequenceScore(
            sequence=sequence,
            residue_log_probabilities=log_probabilities,
            residue_representations=representations,
            pooled_representation=representations.mean(axis=0),
        )


def test_run_esm2_analysis_compares_t2_mutant_with_wt(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    write_fasta('WT', 'AKAA', paths.wt_fasta)
    mutant_path = paths.mutant_fastas / 'K2A.fasta'
    write_fasta('K2A', 'AAAA', mutant_path)
    manifest = pd.DataFrame(
        [
            {
                'variant_id': 'K2A',
                'mutation_spec': 'K2A',
                'fasta_path': str(mutant_path),
                'status': 'PASS',
            }
        ]
    )

    result = run_esm2_analysis(manifest, paths, scorer=FakeESM2Scorer())

    row = result.iloc[0]
    assert row['analysis_status'] == 'PASS'
    assert int(row['mutation_count']) == 1
    assert float(row['delta_pseudo_perplexity']) < 0
    assert (paths.tables / 'ESM2_mutant_comparison.csv').exists()
    per_residue_path = Path(row['per_residue_path'])
    assert per_residue_path.exists()
    per_residue = pd.read_csv(per_residue_path)
    assert per_residue['is_mutation_site'].tolist() == [False, True, False, False]


def test_esm2_analysis_rejects_manifest_sequence_mismatch(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    write_fasta('WT', 'AKAA', paths.wt_fasta)
    mutant_path = paths.mutant_fastas / 'K2A.fasta'
    write_fasta('K2A', 'AAAR', mutant_path)
    manifest = pd.DataFrame(
        [
            {
                'variant_id': 'K2A',
                'mutation_spec': 'K2A',
                'fasta_path': str(mutant_path),
                'status': 'PASS',
            }
        ]
    )

    result = run_esm2_analysis(manifest, paths, scorer=FakeESM2Scorer())

    assert result.iloc[0]['analysis_status'] == 'FAILED'
    assert 'do not match actual WT/mutant differences' in result.iloc[0]['analysis_error']


def test_esm2_config_validates_scoring_weights():
    with pytest.raises(ValueError, match='at least one'):
        ESM2AnalysisConfig(
            perplexity_weight=0.0,
            representation_weight=0.0,
        ).validate()
