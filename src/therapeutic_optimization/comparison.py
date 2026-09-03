from __future__ import annotations

import pandas as pd

from .config import ProjectPaths


def _prediction_summary(predictions: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    if predictions.empty:
        return {
            'lysine_count': 0,
            'positive_site_count': 0,
            'probability_burden': 0.0,
            'mean_probability': 0.0,
            'max_probability': 0.0,
        }
    probabilities = predictions['probability'].astype(float)
    positive = probabilities > threshold
    return {
        'lysine_count': int(len(predictions)),
        'positive_site_count': int(positive.sum()),
        'probability_burden': float(probabilities.loc[positive].sum()),
        'mean_probability': float(probabilities.mean()),
        'max_probability': float(probabilities.max()),
    }


def build_lysine_free_comparison(
    up1: pd.DataFrame,
    ub2: pd.DataFrame,
    manifest: pd.DataFrame,
    s1_metrics: pd.DataFrame,
    paths: ProjectPaths,
) -> pd.DataFrame:
    """Build one audit-friendly WT versus all-K-to-R comparison row."""
    if up1.empty:
        raise ValueError('UP1 is empty; WT ubiquitination metrics cannot be calculated.')
    if len(manifest) != 1:
        raise ValueError('The all-K-to-R comparison requires exactly one mutant manifest row.')

    variant_id = str(manifest.iloc[0]['variant_id'])
    threshold = float(up1['threshold'].iloc[0]) if 'threshold' in up1.columns else 0.40
    mutant_predictions = (
        ub2.loc[ub2['variant_id'].astype(str).eq(variant_id)].copy()
        if 'variant_id' in ub2.columns
        else ub2.iloc[0:0].copy()
    )
    wt = _prediction_summary(up1, threshold)
    mutant = _prediction_summary(mutant_predictions, threshold)

    structural = s1_metrics.loc[s1_metrics['variant_id'].astype(str).eq(variant_id)]
    if structural.empty:
        structural_row: dict = {}
    else:
        structural_row = structural.iloc[0].to_dict()

    record = {
        'variant_id': variant_id,
        'mutation_spec': str(manifest.iloc[0]['mutation_spec']),
        'mutation_count': int(manifest.iloc[0]['mutation_count']),
        'threshold': threshold,
        'wt_lysine_count': wt['lysine_count'],
        'mutant_lysine_count': mutant['lysine_count'],
        'wt_positive_site_count': wt['positive_site_count'],
        'mutant_positive_site_count': mutant['positive_site_count'],
        'wt_positive_probability_burden': wt['probability_burden'],
        'mutant_positive_probability_burden': mutant['probability_burden'],
        'positive_probability_burden_reduction': (
            float(wt['probability_burden']) - float(mutant['probability_burden'])
        ),
        'wt_mean_probability': wt['mean_probability'],
        'mutant_mean_probability': mutant['mean_probability'],
        'wt_max_probability': wt['max_probability'],
        'mutant_max_probability': mutant['max_probability'],
    }
    for column in (
        'analysis_status',
        'analysis_error',
        'structure_pass',
        'structural_failure_reasons',
        'structural_preservation_score',
        'global_ca_rmsd',
        'core_ca_rmsd',
        'binder_ca_rmsd',
        'mean_ca_displacement',
        'local_mean_ca_displacement',
        'mutation_ca_displacement',
        'mutation_ca_displacement_max',
        'radius_of_gyration_change',
        'wt_mean_plddt',
        'mutant_mean_plddt',
        'mean_plddt_change',
        'global_contact_change_fraction',
        'intradomain_contact_change_fraction',
        'global_contacts_lost',
        'global_contacts_gained',
        'local_contacts_lost',
        'local_contacts_gained',
        'wt_structure',
        'mutant_structure',
        'per_residue_csv',
        'displacement_plot',
    ):
        record[column] = structural_row.get(column)

    comparison = pd.DataFrame([record])
    comparison.to_csv(paths.tables / 'LF1_WT_vs_all_K_to_R_comparison.csv', index=False)
    return comparison
