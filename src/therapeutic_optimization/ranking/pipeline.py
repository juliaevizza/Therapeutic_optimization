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
    """R1: record which T2 candidates survived S1 and which dropped out."""
    structural_columns = [
        column
        for column in [
            'variant_id',
            'analysis_status',
            'analysis_error',
            'structure_pass',
            'structural_failure_reasons',
            'structural_preservation_score',
            'global_ca_rmsd',
            'core_ca_rmsd',
            'binder_ca_rmsd',
            'local_mean_ca_displacement',
            'mutation_ca_displacement_max',
            'global_contact_change_fraction',
            'intradomain_contact_change_fraction',
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
    if 'screen_status' in r1.columns:
        r1.loc[r1['screen_status'].eq('ESM2_REJECTED'), 'R1_status'] = 'DROPPED_ESM2'
    r1 = r1.sort_values(
        ['structure_pass', 'structural_preservation_score'],
        ascending=[False, False],
        na_position='last',
    ).reset_index(drop=True)
    r1.to_csv(paths.table('R1_structural_screen.csv'), index=False)
    return r1


def _site_probability_map(df: pd.DataFrame) -> dict[int, float]:
    if df.empty:
        return {}
    return {
        int(row.lysine_position): float(row.probability)
        for row in df.itertuples(index=False)
    }


def attach_esm2_scores(
    ranked: pd.DataFrame,
    esm2_results: pd.DataFrame,
    config: ESM2AnalysisConfig,
) -> pd.DataFrame:
    """Attach bounded ESM-2 compatibility components to R2 candidates.

    The perplexity component is ``min(1, WT_PP / mutant_PP)``.  A mutant that
    is at least as plausible as WT therefore receives 1, while less plausible
    mutants are penalized smoothly.  Pooled cosine similarity is clipped to
    [0, 1] for the representation-preservation component.  Failed or missing
    ESM-2 analyses remain NaN rather than being treated as biological failures.
    """
    config.validate()
    if 'variant_id' not in esm2_results.columns:
        raise ValueError('ESM-2 results are missing required column: variant_id')
    if esm2_results['variant_id'].astype(str).duplicated().any():
        duplicates = sorted(
            esm2_results.loc[
                esm2_results['variant_id'].astype(str).duplicated(keep=False),
                'variant_id',
            ].astype(str).unique()
        )
        raise ValueError(f'ESM-2 results contain duplicate variant IDs: {duplicates}')

    source_to_target = {
        'model_name': 'esm2_model_name',
        'delta_pseudo_log_likelihood': 'esm2_delta_pseudo_log_likelihood',
        'delta_mean_log_probability': 'esm2_delta_mean_log_probability',
        'wt_pseudo_perplexity': 'esm2_wt_pseudo_perplexity',
        'mutant_pseudo_perplexity': 'esm2_mutant_pseudo_perplexity',
        'delta_pseudo_perplexity': 'esm2_delta_pseudo_perplexity',
        'pseudo_perplexity_percent_change': 'esm2_pseudo_perplexity_percent_change',
        'mean_residue_representation_cosine_similarity': (
            'esm2_mean_residue_representation_cosine_similarity'
        ),
        'pooled_representation_cosine_similarity': (
            'esm2_pooled_representation_cosine_similarity'
        ),
        'pooled_representation_cosine_distance': (
            'esm2_pooled_representation_cosine_distance'
        ),
        'mutation_site_representation_cosine_similarity': (
            'esm2_mutation_site_representation_cosine_similarity'
        ),
        'mutation_site_representation_cosine_distance': (
            'esm2_mutation_site_representation_cosine_distance'
        ),
        'analysis_status': 'esm2_analysis_status',
        'analysis_error': 'esm2_analysis_error',
    }
    available_columns = [
        'variant_id',
        *(column for column in source_to_target if column in esm2_results.columns),
    ]
    esm2 = esm2_results[available_columns].copy()
    esm2['variant_id'] = esm2['variant_id'].astype(str)
    esm2 = esm2.rename(columns=source_to_target)
    result = ranked.copy()
    result['variant_id'] = result['variant_id'].astype(str)
    result = result.merge(esm2, on='variant_id', how='left', validate='one_to_one')

    required_metrics = {
        'esm2_delta_mean_log_probability',
        'esm2_pooled_representation_cosine_similarity',
    }
    missing_metrics = required_metrics - set(result.columns)
    if missing_metrics:
        raise ValueError(
            'ESM-2 results are missing scoring columns: '
            f'{sorted(missing_metrics)}'
        )

    delta_mean = pd.to_numeric(
        result['esm2_delta_mean_log_probability'], errors='coerce'
    )
    pooled_cosine = pd.to_numeric(
        result['esm2_pooled_representation_cosine_similarity'], errors='coerce'
    )
    status_pass = (
        result['esm2_analysis_status'].eq('PASS')
        if 'esm2_analysis_status' in result.columns
        else pd.Series(True, index=result.index)
    )
    valid = status_pass & delta_mean.notna() & pooled_cosine.notna()

    result['esm2_perplexity_plausibility_score'] = float('nan')
    result.loc[valid, 'esm2_perplexity_plausibility_score'] = delta_mean.loc[
        valid
    ].map(lambda value: math.exp(min(float(value), 0.0)))
    result['esm2_representation_preservation_score'] = float('nan')
    result.loc[valid, 'esm2_representation_preservation_score'] = pooled_cosine.loc[
        valid
    ].clip(lower=0.0, upper=1.0)

    total_weight = config.perplexity_weight + config.representation_weight
    result['esm2_compatibility_score'] = float('nan')
    result.loc[valid, 'esm2_compatibility_score'] = (
        config.perplexity_weight
        * result.loc[valid, 'esm2_perplexity_plausibility_score']
        + config.representation_weight
        * result.loc[valid, 'esm2_representation_preservation_score']
    ) / total_weight
    result['esm2_score_available'] = valid
    return result


def run_r2(
    up1: pd.DataFrame,
    ub2: pd.DataFrame,
    s1_conserved: pd.DataFrame,
    paths: ProjectPaths,
    esm2_results: pd.DataFrame | None = None,
    esm2_config: ESM2AnalysisConfig | None = None,
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
        ranked.to_csv(paths.table('R2_all_ranked.csv'), index=False)
        ranked.to_csv(paths.table('R2_optimized.csv'), index=False)
        ranked.to_csv(paths.table('R2_needs_further_optimization.csv'), index=False)
        return ranked, ranked.copy(), ranked.copy()

    if esm2_results is not None:
        ranked = attach_esm2_scores(
            ranked,
            esm2_results,
            esm2_config or ESM2AnalysisConfig(),
        )

    group_order = {'optimized': 0, 'needs_further_optimization': 1}
    ranked['_group_order'] = ranked['R2_group'].map(group_order)
    sort_columns = [
        '_group_order',
        'ubiquitination_burden_reduction',
        'mutant_positive_burden',
        'new_positive_count',
    ]
    ascending = [True, False, True, True]
    if esm2_results is not None:
        sort_columns.append('esm2_compatibility_score')
        ascending.append(False)
    sort_columns.append('structural_preservation_score')
    ascending.append(False)
    ranked = ranked.sort_values(
        sort_columns,
        ascending=ascending,
        na_position='last',
    ).drop(columns='_group_order').reset_index(drop=True)
    ranked.insert(0, 'final_rank', range(1, len(ranked) + 1))

    optimized = ranked.loc[ranked['R2_group'].eq('optimized')].copy()
    needs = ranked.loc[ranked['R2_group'].eq('needs_further_optimization')].copy()

    ranked.to_csv(paths.table('R2_all_ranked.csv'), index=False)
    optimized.to_csv(paths.table('R2_optimized.csv'), index=False)
    needs.to_csv(paths.table('R2_needs_further_optimization.csv'), index=False)
    return ranked, optimized, needs
