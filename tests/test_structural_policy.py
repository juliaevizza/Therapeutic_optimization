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


def test_domain_aware_policy_allows_binder_repositioning_but_not_refolding():
    common = {
        'analysis_status': 'PASS',
        'global_ca_rmsd': 6.7,
        'core_ca_rmsd': 0.4,
        'local_mean_ca_displacement': 0.6,
        'mutation_ca_displacement_max': 0.8,
        'global_contact_change_fraction': 0.35,
        'intradomain_contact_change_fraction': 0.03,
        'mutant_mean_plddt': 85.0,
    }
    metrics = pd.DataFrame(
        [
            {'variant_id': 'binder_moved', 'binder_ca_rmsd': 0.5, **common},
            {'variant_id': 'binder_refolded', 'binder_ca_rmsd': 1.5, **common},
        ]
    )
    thresholds = StructuralThresholds(
        binder_start=1,
        binder_end=16,
        binder_ca_rmsd_max=1.0,
    )

    result = classify_structural_preservation(metrics, thresholds)
    status = dict(zip(result['variant_id'], result['structure_pass']))

    assert bool(status['binder_moved']) is True
    assert bool(status['binder_refolded']) is False
    reasons = dict(zip(result['variant_id'], result['structural_failure_reasons']))
    assert reasons['binder_moved'] == ''
    assert reasons['binder_refolded'] == 'BINDER_CONFORMATION_RMSD'
