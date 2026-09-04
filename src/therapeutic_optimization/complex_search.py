"""Ordered, lazy all-site substitution search with ESM-2 and structure gates."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from hashlib import sha256
from itertools import combinations, product
import json
import math

import pandas as pd

from .config import ComplexSearchConfig, ProjectPaths
from .esm2_analysis import SUMMARY_COLUMNS
from .io import apply_mutations, read_single_fasta, write_fasta
from .transformations.mutants import _positive_sites


MANIFEST_COLUMNS = [
    'variant_id', 'mutation_spec', 'mutation_count', 'source_sites',
    'replacement_aas', 'sequence_length', 'fasta_path', 'status', 'error',
    'search_index', 'screen_status',
]
STRUCTURE_COLUMNS = MANIFEST_COLUMNS + [
    'analysis_status', 'analysis_error', 'structure_pass',
    'structural_failure_reasons', 'structural_preservation_score',
]
COSINE_METRIC = 'mean_residue_representation_cosine_similarity'


def ordered_assignments(site_count: int, amino_acids: tuple[str, ...]) -> Iterator[tuple[str, ...]]:
    """Emit every assignment once, by newest amino acid then its multiplicity.

    For two sites and R/Q/E: RR, QR, RQ, QQ, ER, EQ, RE, QE, EE.
    Every slot is always mutated. No full-space list or deduplication set is used.
    """
    if site_count < 0:
        raise ValueError('site_count cannot be negative.')
    if not amino_acids or len(set(amino_acids)) != len(amino_acids):
        raise ValueError('The replacement order must be nonempty and unique.')
    if site_count == 0:
        return
    yield (amino_acids[0],) * site_count
    for rank, newest in enumerate(amino_acids[1:], start=1):
        for count in range(1, site_count + 1):
            for new_positions in combinations(range(site_count), count):
                chosen = set(new_positions)
                other_positions = [i for i in range(site_count) if i not in chosen]
                for others in product(amino_acids[:rank], repeat=len(other_positions)):
                    assignment = [newest] * site_count
                    for position, aa in zip(other_positions, others):
                        assignment[position] = aa
                    yield tuple(assignment)


def esm2_gate(row: pd.Series, config: ComplexSearchConfig) -> tuple[bool, str]:
    """Require the exact mean-residue metric; optionally gate PP change separately."""
    if row.get('analysis_status') != 'PASS':
        return False, 'ESM2_ANALYSIS_FAILED'
    try:
        cosine = float(row[COSINE_METRIC])
    except (KeyError, TypeError, ValueError):
        return False, 'ESM2_COSINE_MISSING'
    if not math.isfinite(cosine) or not cosine > config.min_mean_residue_cosine_similarity:
        return False, 'ESM2_COSINE_GATE'
    if config.max_pseudo_perplexity_percent_change is not None:
        try:
            change = float(row['pseudo_perplexity_percent_change'])
        except (KeyError, TypeError, ValueError):
            return False, 'ESM2_PERPLEXITY_MISSING'
        if not math.isfinite(change) or change > config.max_pseudo_perplexity_percent_change:
            return False, 'ESM2_PERPLEXITY_GATE'
    return True, ''


def run_complex_search(
    up1: pd.DataFrame,
    paths: ProjectPaths,
    config: ComplexSearchConfig,
    threshold: float,
    evaluate_esm2: Callable[[pd.DataFrame], pd.DataFrame],
    evaluate_structure: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[str, object]:
    config.validate()
    paths.ensure()
    _, wt_sequence = read_single_fasta(paths.wt_fasta)
    sites = _positive_sites(up1, threshold)
    for site in sites:
        if not 1 <= site <= len(wt_sequence) or wt_sequence[site - 1] != 'K':
            raise ValueError(f'UP1 selected site {site}, which is not a WT lysine.')

    records: list[dict] = []
    esm_records: list[dict] = []
    structural_records: list[dict] = []
    summary = {
        'status': 'RUNNING', 'target_survivors': config.target_survivors,
        'candidates_evaluated': 0, 'esm2_passed': 0, 'survivors': 0,
        'sites': sites, 'replacement_aas': list(config.replacement_aas),
        'cosine_metric': COSINE_METRIC,
        'strict_cosine_cutoff': config.min_mean_residue_cosine_similarity,
        'max_pseudo_perplexity_percent_change': config.max_pseudo_perplexity_percent_change,
        'max_candidates': config.max_candidates,
        'possible_candidates': len(config.replacement_aas) ** len(sites) if sites else 0,
    }

    def checkpoint() -> dict[str, object]:
        manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
        esm2 = pd.DataFrame(esm_records) if esm_records else pd.DataFrame(
            columns=[*SUMMARY_COLUMNS, 'esm2_screen_pass', 'esm2_failure_reason']
        )
        metrics = pd.DataFrame(structural_records) if structural_records else pd.DataFrame(
            columns=STRUCTURE_COLUMNS
        )
        conserved = metrics.loc[metrics['structure_pass'].eq(True)].copy()
        conserved = conserved.sort_values('structural_preservation_score', ascending=False)
        for name, frame in [
            ('T2_mutation_manifest.csv', manifest), ('ESM2_mutant_comparison.csv', esm2),
            ('S1_structural_metrics.csv', metrics), ('S1_structurally_conserved.csv', conserved),
        ]:
            destination = paths.table(name)
            temporary = destination.with_suffix('.csv.tmp')
            frame.to_csv(temporary, index=False)
            temporary.replace(destination)
        destination = paths.table('T2_search_summary.json')
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        temporary.replace(destination)
        return {'T2': manifest, 'ESM2': esm2, 'S1_metrics': metrics,
                'S1_conserved': conserved, 'search_summary': dict(summary)}

    checkpoint()
    try:
        for assignment in ordered_assignments(len(sites), config.replacement_aas):
            if config.max_candidates is not None and len(records) >= config.max_candidates:
                break
            mutations = tuple(f'K{site}{aa}' for site, aa in zip(sites, assignment))
            mutant_sequence = apply_mutations(wt_sequence, mutations)
            # All-site mutation specifications can exceed filesystem name limits.
            variant_id = 'complex_' + sha256(mutant_sequence.encode()).hexdigest()[:24]
            fasta_path = paths.mutant_fastas / f'{variant_id}.fasta'
            write_fasta(variant_id, mutant_sequence, fasta_path)
            record = dict(zip(MANIFEST_COLUMNS, [
                variant_id, ';'.join(mutations), len(sites),
                ';'.join(f'K{site}' for site in sites), ';'.join(assignment),
                len(mutant_sequence), str(fasta_path), 'PASS', None,
                len(records) + 1, 'PENDING',
            ]))
            records.append(record)
            candidate = pd.DataFrame([record])
            esm2 = evaluate_esm2(candidate)
            if len(esm2) != 1 or str(esm2.iloc[0]['variant_id']) != variant_id:
                raise RuntimeError('ESM2_complex must return the current candidate exactly once.')
            passed, reason = esm2_gate(esm2.iloc[0], config)
            esm_records.append({**esm2.iloc[0].to_dict(), 'esm2_screen_pass': passed,
                                'esm2_failure_reason': reason})
            if esm2.iloc[0]['analysis_status'] != 'PASS':
                raise RuntimeError(f"ESM2_complex failed: {esm2.iloc[0].get('analysis_error')}")
            if passed:
                summary['esm2_passed'] += 1
                metrics, _ = evaluate_structure(candidate)
                if len(metrics) != 1 or str(metrics.iloc[0]['variant_id']) != variant_id:
                    raise RuntimeError('S1_complex must return the current candidate exactly once.')
                structural = {**record, **metrics.iloc[0].to_dict()}
                survived = (structural.get('analysis_status') == 'PASS'
                            and structural.get('structure_pass') == True)
                structural['structure_pass'] = survived
                record['screen_status'] = 'CONSERVED' if survived else 'STRUCTURAL_REJECTED'
                summary['survivors'] += int(survived)
            else:
                record['screen_status'] = 'ESM2_REJECTED'
                structural = {**record, 'analysis_status': 'SKIPPED_ESM2',
                              'analysis_error': None, 'structure_pass': False,
                              'structural_failure_reasons': reason,
                              'structural_preservation_score': float('nan')}
            structural['screen_status'] = record['screen_status']
            structural_records.append(structural)
            summary['candidates_evaluated'] += 1
            checkpoint()
            print(f"T2_complex: evaluated {len(records)}; ESM2 passed {summary['esm2_passed']}; "
                  f"structurally conserved {summary['survivors']}/{config.target_survivors}", flush=True)
            if summary['survivors'] >= config.target_survivors:
                break
    except BaseException as exc:
        summary['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED'
        summary['error'] = str(exc)
        checkpoint()
        raise

    summary['status'] = (
        'TARGET_REACHED' if summary['survivors'] >= config.target_survivors else
        'NO_PROBLEMATIC_SITES' if not sites else
        'EXHAUSTED' if len(records) == summary['possible_candidates'] else 'CANDIDATE_LIMIT'
    )
    print(f"T2_complex finished: {summary['status']} ({summary['survivors']} survivors).", flush=True)
    return checkpoint()
