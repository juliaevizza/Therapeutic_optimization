from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from ..config import ProjectPaths, StructuralThresholds, StructurePredictorConfig
from .colabfold import ColabFoldPredictor, StructurePrediction, find_rank1_structure
from .metrics import analyze_structure_pair


def build_structure_predictor(config: StructurePredictorConfig) -> ColabFoldPredictor:
    if config.name.lower() in {'colabfold', 'alphafold', 'alphafold2'}:
        return ColabFoldPredictor(config.executable, config.extra_args)
    raise ValueError(
        f'Unknown structure predictor {config.name!r}. Add an adapter and register it here.'
    )


def _preservation_score(row: pd.Series, thresholds: StructuralThresholds) -> float:
    domain_aware = thresholds.binder_positions() is not None
    rmsd = row['core_ca_rmsd'] if domain_aware else row['global_ca_rmsd']
    contacts = (
        row['intradomain_contact_change_fraction']
        if domain_aware else row['global_contact_change_fraction']
    )
    penalties = [
        min(
            float(rmsd)
            / (thresholds.core_ca_rmsd_max if domain_aware else thresholds.global_ca_rmsd_max),
            2.0,
        ),
        *(
            [min(float(row['binder_ca_rmsd']) / thresholds.binder_ca_rmsd_max, 2.0)]
            if domain_aware else []
        ),
        min(float(row['local_mean_ca_displacement']) / thresholds.local_mean_ca_displacement_max, 2.0),
        min(float(row['mutation_ca_displacement_max']) / thresholds.mutation_ca_displacement_max, 2.0),
        min(float(contacts) / thresholds.contact_change_fraction_max, 2.0),
    ]
    geometry_score = max(0.0, 1.0 - float(np.mean(penalties)) / 2.0)
    confidence_score = min(1.0, float(row['mutant_mean_plddt']) / 100.0)
    return float(0.8 * geometry_score + 0.2 * confidence_score)


def classify_structural_preservation(
    metrics: pd.DataFrame,
    thresholds: StructuralThresholds,
) -> pd.DataFrame:
    result = metrics.copy()
    if result.empty:
        result['structure_pass'] = pd.Series(dtype=bool)
        result['structural_failure_reasons'] = pd.Series(dtype=str)
        result['structural_preservation_score'] = pd.Series(dtype=float)
        return result

    binder_range = thresholds.binder_positions()
    domain_aware = binder_range is not None
    metric_columns = [
        'global_ca_rmsd',
        'local_mean_ca_displacement',
        'mutation_ca_displacement_max',
        'global_contact_change_fraction',
        'mutant_mean_plddt',
        'core_ca_rmsd',
        'binder_ca_rmsd',
        'intradomain_contact_change_fraction',
    ]
    for column in metric_columns:
        if column not in result.columns:
            result[column] = np.nan

    success = result['analysis_status'].eq('PASS')
    rmsd_pass = (
        result['core_ca_rmsd'] <= thresholds.core_ca_rmsd_max
        if domain_aware else result['global_ca_rmsd'] <= thresholds.global_ca_rmsd_max
    )
    binder_pass = (
        result['binder_ca_rmsd'] <= thresholds.binder_ca_rmsd_max
        if domain_aware else True
    )
    contact_pass = (
        result['intradomain_contact_change_fraction'] <= thresholds.contact_change_fraction_max
        if domain_aware else result['global_contact_change_fraction'] <= thresholds.contact_change_fraction_max
    )
    result['structure_pass'] = (
        success
        & rmsd_pass
        & binder_pass
        & (result['local_mean_ca_displacement'] <= thresholds.local_mean_ca_displacement_max)
        & (result['mutation_ca_displacement_max'] <= thresholds.mutation_ca_displacement_max)
        & contact_pass
        & (result['mutant_mean_plddt'] >= thresholds.min_mean_plddt)
    )

    def failure_reasons(row: pd.Series) -> str:
        if row['analysis_status'] != 'PASS':
            return 'ANALYSIS_FAILED'
        reasons: list[str] = []
        if domain_aware:
            if row['core_ca_rmsd'] > thresholds.core_ca_rmsd_max:
                reasons.append('CORE_RMSD')
            if row['binder_ca_rmsd'] > thresholds.binder_ca_rmsd_max:
                reasons.append('BINDER_CONFORMATION_RMSD')
            if row['intradomain_contact_change_fraction'] > thresholds.contact_change_fraction_max:
                reasons.append('INTRADOMAIN_CONTACT_CHANGE')
        else:
            if row['global_ca_rmsd'] > thresholds.global_ca_rmsd_max:
                reasons.append('GLOBAL_RMSD')
            if row['global_contact_change_fraction'] > thresholds.contact_change_fraction_max:
                reasons.append('GLOBAL_CONTACT_CHANGE')
        if row['local_mean_ca_displacement'] > thresholds.local_mean_ca_displacement_max:
            reasons.append('MUTATION_REGION_MEAN_DISPLACEMENT')
        if row['mutation_ca_displacement_max'] > thresholds.mutation_ca_displacement_max:
            reasons.append('MUTATION_SITE_MAX_DISPLACEMENT')
        if row['mutant_mean_plddt'] < thresholds.min_mean_plddt:
            reasons.append('LOW_MEAN_PLDDT')
        return ';'.join(reasons)

    result['structural_failure_reasons'] = result.apply(failure_reasons, axis=1)
    result['structural_preservation_score'] = result.apply(
        lambda row: _preservation_score(row, thresholds) if row['analysis_status'] == 'PASS' else 0.0,
        axis=1,
    )
    return result


def run_s1(
    manifest: pd.DataFrame,
    paths: ProjectPaths,
    predictor: ColabFoldPredictor,
    thresholds: StructuralThresholds,
    predict_structures: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    S1: predict WT/mutant structures, compute metrics, and emit the conserved subset.
    """
    paths.ensure()
    valid_manifest = manifest.loc[manifest['status'].eq('PASS')].copy()

    if predict_structures:
        predictions = [
            StructurePrediction('WT', paths.wt_fasta, paths.structures_wt),
            *(
                StructurePrediction(
                    str(row.variant_id),
                    Path(row.fasta_path),
                    paths.structures_mutants / str(row.variant_id),
                )
                for row in valid_manifest.itertuples(index=False)
            ),
        ]
        structures = predictor.predict_batch(
            predictions,
            paths.storage / 'structures' / 'batch',
        )
        wt_structure = structures['WT']
    else:
        wt_structure = find_rank1_structure(paths.structures_wt)

    records: list[dict] = []
    for row in valid_manifest.itertuples(index=False):
        variant_id = str(row.variant_id)
        mutant_output = paths.structures_mutants / variant_id
        try:
            if predict_structures:
                mutant_structure = structures[variant_id]
            else:
                mutant_structure = find_rank1_structure(mutant_output)
            metrics = analyze_structure_pair(
                wt_structure_path=wt_structure,
                mutant_structure_path=mutant_structure,
                variant_id=variant_id,
                mutation_spec=str(row.mutation_spec),
                per_residue_dir=paths.per_residue,
                figure_dir=paths.figures,
                binder_range=thresholds.binder_positions(),
            )
            metrics['analysis_status'] = 'PASS'
            metrics['analysis_error'] = None
        except Exception as exc:
            metrics = {
                'variant_id': variant_id,
                'mutation_spec': str(row.mutation_spec),
                'mutation_count': int(row.mutation_count),
                'analysis_status': 'FAILED',
                'analysis_error': str(exc),
                'wt_structure': str(wt_structure),
                'mutant_structure': None,
            }
        records.append(metrics)

    raw = pd.DataFrame(records)
    if not raw.empty:
        raw = valid_manifest.merge(raw, on=['variant_id', 'mutation_spec', 'mutation_count'], how='left')
    else:
        raw = valid_manifest.copy()
        raw['analysis_status'] = pd.Series(dtype=str)
        raw['analysis_error'] = pd.Series(dtype=str)

    classified = classify_structural_preservation(raw, thresholds)
    classified.to_csv(paths.tables / 'S1_structural_metrics.csv', index=False)
    conserved = classified.loc[classified['structure_pass']].copy()
    conserved = conserved.sort_values('structural_preservation_score', ascending=False)
    conserved.to_csv(paths.tables / 'S1_structurally_conserved.csv', index=False)
    return classified, conserved
