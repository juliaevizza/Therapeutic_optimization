from __future__ import annotations

import math

import pandas as pd

from ..config import ProjectPaths
from ..io import parse_mutation


def run_r1(
    t2_manifest: pd.DataFrame,
    s1_metrics: pd.DataFrame,
    paths: ProjectPaths,
) -> pd.DataFrame:
    """R1: record which T2 candidates survived S1 and which dropped out."""
    structural_columns = [
        column
        for column in [
            'variant_id',
            'analysis_status',
            'analysis_error',
            'structure_pass',
            'structural_preservation_score',
            'global_ca_rmsd',
            'local_mean_ca_displacement',
            'mutation_ca_displacement_max',
            'global_contact_change_fraction',
            'mutant_mean_plddt',
        ]
        if column in s1_metrics.columns
    ]
    structural = s1_metrics[structural_columns].drop_duplicates('variant_id')
    r1 = t2_manifest.merge(structural, on='variant_id', how='left')
    r1['structure_pass'] = r1['structure_pass'].fillna(False).astype(bool)
    r1['dropped_after_S1'] = ~r1['structure_pass']
    r1['R1_status'] = r1['structure_pass'].map(
        {True: 'ADVANCE_TO_UB2', False: 'DROPPED_STRUCTURAL'}
    )
    r1 = r1.sort_values(
        ['structure_pass', 'structural_preservation_score'],
        ascending=[False, False],
        na_position='last',
    ).reset_index(drop=True)
    r1.to_csv(paths.tables / 'R1_structural_screen.csv', index=False)
    return r1


def _site_probability_map(df: pd.DataFrame) -> dict[int, float]:
    if df.empty:
        return {}
    return {
        int(row.lysine_position): float(row.probability)
        for row in df.itertuples(index=False)
    }


def run_r2(
    up1: pd.DataFrame,
    ub2: pd.DataFrame,
    s1_conserved: pd.DataFrame,
    paths: ProjectPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    R2: compare WT and mutant ubiquitination predictions.

    optimized = no mutant lysine remains at/above the predictor threshold.
    needs_further_optimization = at least one positive lysine remains.
    """
    if up1.empty:
        raise ValueError('UP1 is empty; R2 has no WT ubiquitination baseline to compare against.')

    threshold = float(up1['threshold'].iloc[0]) if 'threshold' in up1.columns else 0.40
    wt_positive = up1.loc[up1['probability'].astype(float) > threshold].copy()
    wt_positive_map = _site_probability_map(wt_positive)
    wt_positive_sites = set(wt_positive_map)
    wt_positive_burden = float(sum(wt_positive_map.values()))

    records: list[dict] = []
    for row in s1_conserved.itertuples(index=False):
        variant_id = str(row.variant_id)
        mutation_spec = str(row.mutation_spec)
        mutations = [item for item in mutation_spec.split(';') if item]
        targeted_positions = [parse_mutation(item)[1] for item in mutations]

        mutant_rows = ub2.loc[ub2['variant_id'].astype(str) == variant_id].copy()
        mutant_positive = mutant_rows.loc[
            mutant_rows['probability'].astype(float) > threshold
        ].copy()
        mutant_map = _site_probability_map(mutant_positive)
        mutant_positive_sites = set(mutant_map)

        targeted_wt_probability_sum = float(
            sum(wt_positive_map.get(position, 0.0) for position in targeted_positions)
        )
        mutant_positive_burden = float(sum(mutant_map.values()))
        burden_reduction = wt_positive_burden - mutant_positive_burden
        burden_reduction_fraction = (
            burden_reduction / wt_positive_burden if wt_positive_burden > 0 else 0.0
        )

        new_sites = sorted(mutant_positive_sites - wt_positive_sites)
        remaining_wt_sites = sorted(mutant_positive_sites & wt_positive_sites)
        removed_wt_sites = sorted(wt_positive_sites - mutant_positive_sites)

        group = (
            'optimized'
            if len(mutant_positive_sites) == 0
            else 'needs_further_optimization'
        )

        records.append(
            {
                'variant_id': variant_id,
                'mutation_spec': mutation_spec,
                'mutation_count': int(row.mutation_count),
                'structural_preservation_score': float(row.structural_preservation_score),
                'wt_positive_count': len(wt_positive_sites),
                'mutant_positive_count': len(mutant_positive_sites),
                'targeted_wt_probability_sum': targeted_wt_probability_sum,
                'wt_positive_burden': wt_positive_burden,
                'mutant_positive_burden': mutant_positive_burden,
                'ubiquitination_burden_reduction': burden_reduction,
                'ubiquitination_burden_reduction_fraction': burden_reduction_fraction,
                'removed_wt_sites': ';'.join(f'K{p}' for p in removed_wt_sites),
                'remaining_wt_positive_sites': ';'.join(f'K{p}' for p in remaining_wt_sites),
                'new_positive_sites': ';'.join(f'K{p}' for p in new_sites),
                'new_positive_count': len(new_sites),
                'R2_group': group,
            }
        )

    ranked = pd.DataFrame(records)
    if ranked.empty:
        ranked.to_csv(paths.tables / 'R2_all_ranked.csv', index=False)
        ranked.to_csv(paths.tables / 'R2_optimized.csv', index=False)
        ranked.to_csv(paths.tables / 'R2_needs_further_optimization.csv', index=False)
        return ranked, ranked.copy(), ranked.copy()

    group_order = {'optimized': 0, 'needs_further_optimization': 1}
    ranked['_group_order'] = ranked['R2_group'].map(group_order)
    ranked = ranked.sort_values(
        [
            '_group_order',
            'ubiquitination_burden_reduction',
            'mutant_positive_burden',
            'new_positive_count',
            'structural_preservation_score',
        ],
        ascending=[True, False, True, True, False],
    ).drop(columns='_group_order').reset_index(drop=True)
    ranked.insert(0, 'final_rank', range(1, len(ranked) + 1))

    optimized = ranked.loc[ranked['R2_group'].eq('optimized')].copy()
    needs = ranked.loc[ranked['R2_group'].eq('needs_further_optimization')].copy()

    ranked.to_csv(paths.tables / 'R2_all_ranked.csv', index=False)
    optimized.to_csv(paths.tables / 'R2_optimized.csv', index=False)
    needs.to_csv(paths.tables / 'R2_needs_further_optimization.csv', index=False)
    return ranked, optimized, needs
