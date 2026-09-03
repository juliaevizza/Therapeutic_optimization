from __future__ import annotations

import pandas as pd

from therapeutic_optimization.config import ESM2AnalysisConfig, ProjectPaths
from therapeutic_optimization.ranking import run_r1, run_r2


def test_r1_marks_structural_dropouts(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    t2 = pd.DataFrame(
        [
            {'variant_id': 'K2A', 'status': 'PASS', 'mutation_spec': 'K2A'},
            {'variant_id': 'K5A', 'status': 'PASS', 'mutation_spec': 'K5A'},
        ]
    )
    s1 = pd.DataFrame(
        [
            {'variant_id': 'K2A', 'structure_pass': True, 'structural_preservation_score': 0.9, 'analysis_status': 'PASS'},
            {'variant_id': 'K5A', 'structure_pass': False, 'structural_preservation_score': 0.3, 'analysis_status': 'PASS'},
        ]
    )
    result = run_r1(t2, s1, paths)
    status = dict(zip(result['variant_id'], result['R1_status']))
    assert status['K2A'] == 'ADVANCE_TO_UB2'
    assert status['K5A'] == 'DROPPED_STRUCTURAL'


def test_r2_detects_new_sites_and_optimized_group(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    up1 = pd.DataFrame(
        [
            {'variant_id': 'WT', 'lysine_position': 2, 'probability': 0.90, 'threshold': 0.40, 'is_positive': True},
            {'variant_id': 'WT', 'lysine_position': 5, 'probability': 0.60, 'threshold': 0.40, 'is_positive': True},
            {'variant_id': 'WT', 'lysine_position': 8, 'probability': 0.20, 'threshold': 0.40, 'is_positive': False},
        ]
    )
    s1 = pd.DataFrame(
        [
            {'variant_id': 'K2A', 'mutation_spec': 'K2A', 'mutation_count': 1, 'structural_preservation_score': 0.95},
            {'variant_id': 'K2A__K5A', 'mutation_spec': 'K2A;K5A', 'mutation_count': 2, 'structural_preservation_score': 0.85},
        ]
    )
    ub2 = pd.DataFrame(
        [
            {'variant_id': 'K2A', 'lysine_position': 5, 'probability': 0.55},
            {'variant_id': 'K2A', 'lysine_position': 8, 'probability': 0.50},
            {'variant_id': 'K2A__K5A', 'lysine_position': 8, 'probability': 0.10},
        ]
    )
    ranked, optimized, needs = run_r2(up1, ub2, s1, paths)
    assert optimized['variant_id'].tolist() == ['K2A__K5A']
    k2a = needs.loc[needs['variant_id'].eq('K2A')].iloc[0]
    assert k2a['new_positive_sites'] == 'K8'
    assert k2a['R2_group'] == 'needs_further_optimization'
    assert ranked.iloc[0]['variant_id'] == 'K2A__K5A'


def test_r2_uses_esm2_compatibility_to_break_therapeutically_equal_candidates(
    tmp_path,
):
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    up1 = pd.DataFrame(
        [
            {
                'variant_id': 'WT',
                'lysine_position': 2,
                'probability': 0.90,
                'threshold': 0.40,
            }
        ]
    )
    s1 = pd.DataFrame(
        [
            {
                'variant_id': 'K2A',
                'mutation_spec': 'K2A',
                'mutation_count': 1,
                'structural_preservation_score': 0.99,
            },
            {
                'variant_id': 'K2R',
                'mutation_spec': 'K2R',
                'mutation_count': 1,
                'structural_preservation_score': 0.80,
            },
        ]
    )
    ub2 = pd.DataFrame(columns=['variant_id', 'lysine_position', 'probability'])
    esm2 = pd.DataFrame(
        [
            {
                'variant_id': 'K2A',
                'model_name': 'fake/esm2',
                'delta_mean_log_probability': -2.0,
                'pooled_representation_cosine_similarity': 0.50,
                'analysis_status': 'PASS',
            },
            {
                'variant_id': 'K2R',
                'model_name': 'fake/esm2',
                'delta_mean_log_probability': 0.10,
                'pooled_representation_cosine_similarity': 0.99,
                'analysis_status': 'PASS',
            },
        ]
    )

    ranked, _optimized, _needs = run_r2(
        up1,
        ub2,
        s1,
        paths,
        esm2_results=esm2,
        esm2_config=ESM2AnalysisConfig(
            perplexity_weight=0.70,
            representation_weight=0.30,
        ),
    )

    assert ranked['variant_id'].tolist() == ['K2R', 'K2A']
    assert ranked['esm2_score_available'].all()
    assert ranked.iloc[0]['esm2_compatibility_score'] > ranked.iloc[1][
        'esm2_compatibility_score'
    ]
