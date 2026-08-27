from __future__ import annotations

import pandas as pd

from therapeutic_optimization.config import StructuralThresholds
from therapeutic_optimization.structural_analysis.pipeline import classify_structural_preservation


def test_structural_policy_pass_and_fail():
    metrics = pd.DataFrame(
        [
            {
                'variant_id': 'good',
                'analysis_status': 'PASS',
                'global_ca_rmsd': 0.2,
                'local_mean_ca_displacement': 0.3,
                'mutation_ca_displacement_max': 0.5,
                'global_contact_change_fraction': 0.02,
                'mutant_mean_plddt': 90.0,
            },
            {
                'variant_id': 'bad',
                'analysis_status': 'PASS',
                'global_ca_rmsd': 1.5,
                'local_mean_ca_displacement': 0.3,
                'mutation_ca_displacement_max': 0.5,
                'global_contact_change_fraction': 0.02,
                'mutant_mean_plddt': 90.0,
            },
        ]
    )
    result = classify_structural_preservation(metrics, StructuralThresholds())
    status = dict(zip(result['variant_id'], result['structure_pass']))
    assert bool(status['good']) is True
    assert bool(status['bad']) is False
