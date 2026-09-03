from __future__ import annotations

import pandas as pd
import pytest

from therapeutic_optimization.config import MutationConfig, ProjectPaths
from therapeutic_optimization.io import apply_mutations, normalize_sequence, read_single_fasta
from therapeutic_optimization.transformations import (
    generate_all_lysine_to_arginine_manifest,
    generate_mutant_manifest,
    prepare_wt_input,
)


def fake_up1() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {'lysine_position': 2, 'probability': 0.90, 'is_positive': True},
            {'lysine_position': 5, 'probability': 0.65, 'is_positive': True},
            {'lysine_position': 8, 'probability': 0.20, 'is_positive': False},
        ]
    )


def test_normalize_sequence():
    assert normalize_sequence(' acd\nEFG ') == 'ACDEFG'
    with pytest.raises(ValueError):
        normalize_sequence('ACDX')


def test_apply_mutations_validates_wt_residue():
    sequence = 'AKAAKAAA'
    assert apply_mutations(sequence, ['K2A']) == 'AAAAKAAA'
    assert apply_mutations(sequence, ['K2R', 'K5A']) == 'ARA AAAAA'.replace(' ', '')
    with pytest.raises(ValueError):
        apply_mutations(sequence, ['A2R'])


def test_single_mutant_generation(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    prepare_wt_input('AKAAKAAK', 'WT', paths)
    manifest = generate_mutant_manifest(
        fake_up1(),
        paths,
        MutationConfig(mode='single', replacement_aas=('A',), threshold=0.40),
    )
    assert manifest['variant_id'].tolist() == ['K2A', 'K5A']
    assert manifest['status'].eq('PASS').all()
    _, sequence = read_single_fasta(paths.mutant_fastas / 'K2A.fasta')
    assert sequence == 'AAAAKAAK'


def test_combinatorial_generation(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    prepare_wt_input('AKAAKAAK', 'WT', paths)
    manifest = generate_mutant_manifest(
        fake_up1(),
        paths,
        MutationConfig(
            mode='combinatorial',
            replacement_aas=('A', 'R'),
            max_combination_order=2,
            max_variants=100,
            threshold=0.40,
        ),
    )
    # 2 sites * 2 replacements + C(2,2) * 2^2 = 8 variants
    assert len(manifest) == 8
    assert 'K2A__K5R' in set(manifest['variant_id'])


def test_variant_guard(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    prepare_wt_input('AKAAKAAK', 'WT', paths)
    with pytest.raises(RuntimeError):
        generate_mutant_manifest(
            fake_up1(),
            paths,
            MutationConfig(
                mode='combinatorial',
                replacement_aas=('A', 'R'),
                max_combination_order=2,
                max_variants=3,
            ),
        )


def test_generate_all_lysine_to_arginine_manifest(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    prepare_wt_input('AKAAKAAK', 'WT', paths)
    manifest = generate_all_lysine_to_arginine_manifest(paths)

    assert manifest['variant_id'].tolist() == ['ALL_K_TO_R']
    assert manifest.iloc[0]['mutation_spec'] == 'K2R;K5R;K8R'
    assert int(manifest.iloc[0]['mutation_count']) == 3
    _, sequence = read_single_fasta(paths.mutant_fastas / 'ALL_K_TO_R.fasta')
    assert sequence == 'ARAARAAR'


def test_all_lysine_to_arginine_rejects_already_lysine_free_wt(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    prepare_wt_input('ARAARAAR', 'WT', paths)
    with pytest.raises(ValueError, match='already lysine-free'):
        generate_all_lysine_to_arginine_manifest(paths)
