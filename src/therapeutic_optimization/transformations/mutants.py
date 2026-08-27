from __future__ import annotations

from itertools import combinations, product
from math import prod
from pathlib import Path

import pandas as pd

from ..config import MutationConfig, ProjectPaths
from ..io import apply_mutations, read_single_fasta, write_fasta

REQUIRED_UP1_COLUMNS = {'lysine_position', 'probability'}


def _positive_sites(up1: pd.DataFrame, threshold: float) -> list[int]:
    missing = REQUIRED_UP1_COLUMNS - set(up1.columns)
    if missing:
        raise ValueError(f'UP1 is missing required columns: {sorted(missing)}')

    positions = (
        up1.loc[
            up1['probability'].astype(float) > threshold,
            'lysine_position',
        ]
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return positions


def _candidate_specs(
    sites: list[int],
    config: MutationConfig,
) -> list[tuple[str, ...]]:
    config.validate()
    replacement_aas = tuple(aa.upper() for aa in config.replacement_aas)

    specs: list[tuple[str, ...]] = []
    if config.mode == 'single':
        for position in sites:
            for replacement in replacement_aas:
                specs.append((f'K{position}{replacement}',))
        return specs

    max_order = min(config.max_combination_order, len(sites))
    for order in range(1, max_order + 1):
        for site_combo in combinations(sites, order):
            for aa_combo in product(replacement_aas, repeat=order):
                specs.append(
                    tuple(
                        f'K{position}{replacement}'
                        for position, replacement in zip(site_combo, aa_combo)
                    )
                )
                if len(specs) > config.max_variants:
                    raise RuntimeError(
                        'Requested mutation space exceeds max_variants='
                        f'{config.max_variants}. Reduce replacement_aas or '
                        'max_combination_order, or deliberately raise the guard.'
                    )
    return specs


def estimate_variant_count(site_count: int, config: MutationConfig) -> int:
    """Estimate the size of T2 before generating FASTAs."""
    from math import comb

    r = len(config.replacement_aas)
    if config.mode == 'single':
        return site_count * r
    max_order = min(config.max_combination_order, site_count)
    return sum(comb(site_count, order) * (r ** order) for order in range(1, max_order + 1))


def generate_mutant_manifest(
    up1: pd.DataFrame,
    paths: ProjectPaths,
    config: MutationConfig,
) -> pd.DataFrame:
    """
    T2: turn positive WT ubiquitination sites into mutant FASTAs.

    single mode: one positive lysine is removed per variant.
    combinatorial mode: generates orders 1..max_combination_order.
    """
    config.validate()
    paths.ensure()
    protein_id, wt_sequence = read_single_fasta(paths.wt_fasta)
    sites = _positive_sites(up1, config.threshold)

    if not sites:
        columns = [
            'variant_id', 'mutation_spec', 'mutation_count', 'source_sites',
            'replacement_aas', 'sequence_length', 'fasta_path', 'status', 'error',
        ]
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(paths.tables / 'T2_mutation_manifest.csv', index=False)
        return empty

    estimated = estimate_variant_count(len(sites), config)
    if estimated > config.max_variants:
        raise RuntimeError(
            f'T2 would generate {estimated} variants, above max_variants={config.max_variants}.'
        )

    records: list[dict] = []
    for mutation_tuple in _candidate_specs(sites, config):
        variant_id = '__'.join(mutation_tuple)
        fasta_path = paths.mutant_fastas / f'{variant_id}.fasta'
        try:
            mutant_sequence = apply_mutations(wt_sequence, mutation_tuple)
            write_fasta(variant_id, mutant_sequence, fasta_path)
            records.append(
                {
                    'variant_id': variant_id,
                    'mutation_spec': ';'.join(mutation_tuple),
                    'mutation_count': len(mutation_tuple),
                    'source_sites': ';'.join(m[:-1] for m in mutation_tuple),
                    'replacement_aas': ';'.join(m[-1] for m in mutation_tuple),
                    'sequence_length': len(mutant_sequence),
                    'fasta_path': str(fasta_path),
                    'status': 'PASS',
                    'error': None,
                }
            )
        except Exception as exc:
            records.append(
                {
                    'variant_id': variant_id,
                    'mutation_spec': ';'.join(mutation_tuple),
                    'mutation_count': len(mutation_tuple),
                    'source_sites': ';'.join(m[:-1] for m in mutation_tuple),
                    'replacement_aas': ';'.join(m[-1] for m in mutation_tuple),
                    'sequence_length': len(wt_sequence),
                    'fasta_path': str(fasta_path),
                    'status': 'FAILED',
                    'error': str(exc),
                }
            )

    manifest = pd.DataFrame(records)
    manifest.to_csv(paths.tables / 'T2_mutation_manifest.csv', index=False)

    failures = manifest['status'].eq('FAILED').sum()
    if failures:
        raise RuntimeError(
            f'{failures} T2 mutation(s) failed validation. '
            'See storage/tables/T2_mutation_manifest.csv.'
        )
    return manifest
