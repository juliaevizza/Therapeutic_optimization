from __future__ import annotations

import pandas as pd

from therapeutic_optimization.comparison import build_lysine_free_comparison
from therapeutic_optimization.config import ProjectPaths
from therapeutic_optimization.ubiquitination_prediction.base import STANDARD_COLUMNS
from therapeutic_optimization.workflow import OptimizationWorkflow


def test_build_lysine_free_comparison(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    up1 = pd.DataFrame(
        [
            {'variant_id': 'WT', 'lysine_position': 2, 'probability': 0.9, 'threshold': 0.4},
            {'variant_id': 'WT', 'lysine_position': 5, 'probability': 0.3, 'threshold': 0.4},
        ]
    )
    ub2 = pd.DataFrame(
        columns=['variant_id', 'lysine_position', 'probability', 'threshold']
    )
    manifest = pd.DataFrame(
        [
            {
                'variant_id': 'ALL_K_TO_R',
                'mutation_spec': 'K2R;K5R',
                'mutation_count': 2,
            }
        ]
    )
    s1 = pd.DataFrame(
        [
            {
                'variant_id': 'ALL_K_TO_R',
                'analysis_status': 'PASS',
                'structure_pass': True,
                'structural_preservation_score': 0.91,
                'global_ca_rmsd': 0.2,
            }
        ]
    )

    result = build_lysine_free_comparison(up1, ub2, manifest, s1, paths)
    row = result.iloc[0]
    assert int(row['wt_lysine_count']) == 2
    assert int(row['mutant_lysine_count']) == 0
    assert int(row['wt_positive_site_count']) == 1
    assert float(row['wt_positive_probability_burden']) == 0.9
    assert float(row['mutant_positive_probability_burden']) == 0.0
    assert bool(row['structure_pass']) is True
    assert float(row['global_ca_rmsd']) == 0.2


class FakeUbiquitinationPredictor:
    name = 'fake'
    threshold = 0.4

    def predict_sequence(self, sequence, protein_id, variant_id, output_dir=None):
        rows = [
            {
                'variant_id': variant_id,
                'protein_id': protein_id,
                'predictor': self.name,
                'lysine_position': position,
                'site': f'K{position}',
                'sequence_context': sequence,
                'probability': 0.8,
                'threshold': self.threshold,
                'is_positive': True,
            }
            for position, residue in enumerate(sequence, start=1)
            if residue == 'K'
        ]
        return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def test_lysine_free_workflow_runs_mutant_scoring_even_after_structural_failure(
    tmp_path,
    monkeypatch,
):
    workflow = OptimizationWorkflow(tmp_path)
    workflow._ubi_predictor = FakeUbiquitinationPredictor()

    def fake_s1(self, manifest, predict_structures=True):
        metrics = manifest.copy()
        metrics['analysis_status'] = 'PASS'
        metrics['analysis_error'] = None
        metrics['structure_pass'] = False
        metrics['structural_preservation_score'] = 0.25
        metrics['global_ca_rmsd'] = 2.0
        metrics['local_mean_ca_displacement'] = 2.0
        metrics['mutation_ca_displacement_max'] = 3.0
        metrics['global_contact_change_fraction'] = 0.2
        metrics['mutant_mean_plddt'] = 80.0
        return metrics, metrics.iloc[0:0].copy()

    monkeypatch.setattr(OptimizationWorkflow, 'S1', fake_s1)
    results = workflow.run_lysine_free_comparison(
        sequence='AKAAK',
        protein_id='WT',
    )

    assert int(results['T2_all_lysine_to_arginine'].iloc[0]['mutation_count']) == 2
    assert results['UB2'].empty
    comparison = results['lysine_free_comparison'].iloc[0]
    assert int(comparison['wt_positive_site_count']) == 2
    assert int(comparison['mutant_positive_site_count']) == 0
    assert bool(comparison['structure_pass']) is False
