from itertools import islice, product
import json
from pathlib import Path

import pandas as pd
import pytest

from therapeutic_optimization.complex_search import (
    COSINE_METRIC, esm2_gate, ordered_assignments, run_complex_search,
)
from therapeutic_optimization.config import ComplexSearchConfig, ProjectPaths, WorkflowConfig
from therapeutic_optimization.io import read_single_fasta, write_fasta
from therapeutic_optimization.workflow import OptimizationWorkflow


def test_ordered_panning_and_complete_space():
    assert list(ordered_assignments(2, ('R', 'H', 'Q'))) == [
        ('R', 'R'), ('H', 'R'), ('R', 'H'), ('H', 'H'),
        ('Q', 'R'), ('Q', 'H'), ('R', 'Q'), ('H', 'Q'), ('Q', 'Q'),
    ]
    aas = ComplexSearchConfig().replacement_aas
    assert aas == ('R', 'H', 'Q', 'E', 'C')
    assignments = list(ordered_assignments(3, aas))
    assert len(assignments) == len(set(assignments)) == 125
    assert set(assignments) == set(product(aas, repeat=3))
    assert assignments[0] == ('R',) * 3
    assert assignments[-1] == ('C',) * 3
    # Huge spaces can produce a prefix immediately, without materialization.
    assert len(list(islice(ordered_assignments(100, aas), 3))) == 3
    assert list(ordered_assignments(0, aas)) == []


@pytest.mark.parametrize('cosine, expected', [
    (.9995, False), (.9994999, False), (.9995001, True),
    (float('nan'), False), (float('inf'), False),
])
def test_strict_exact_metric_gate(cosine, expected):
    row = pd.Series({'analysis_status': 'PASS', COSINE_METRIC: cosine,
                     'pooled_representation_cosine_similarity': 1.0,
                     'pseudo_perplexity_percent_change': 999.0})
    assert esm2_gate(row, ComplexSearchConfig())[0] is expected


def test_perplexity_is_separate_and_optional():
    row = pd.Series({'analysis_status': 'PASS', COSINE_METRIC: 1.0,
                     'pseudo_perplexity_percent_change': 0.1})
    assert esm2_gate(row, ComplexSearchConfig())[0]
    assert not esm2_gate(row, ComplexSearchConfig(max_pseudo_perplexity_percent_change=0))[0]
    row['pseudo_perplexity_percent_change'] = -1
    assert esm2_gate(row, ComplexSearchConfig(max_pseudo_perplexity_percent_change=0))[0]
    row['analysis_status'] = 'FAILED'
    assert not esm2_gate(row, ComplexSearchConfig())[0]


def setup_search(tmp_path, sequence='KAKK'):
    paths = ProjectPaths.from_root(tmp_path, 'complex')
    paths.ensure()
    write_fasta('WT', sequence, paths.wt_fasta)
    # K3 is exactly on the threshold and must remain WT. K1 is duplicated.
    up1 = pd.DataFrame({'lysine_position': [4, 1, 3, 1], 'probability': [.9, .8, .4, .7]})
    return paths, up1


def passing_esm(candidate):
    return pd.DataFrame([{'variant_id': candidate.iloc[0].variant_id,
                          'analysis_status': 'PASS', COSINE_METRIC: .9999}])


def passing_structure(candidate):
    metrics = candidate.assign(analysis_status='PASS', structure_pass=True,
                               structural_preservation_score=.95)
    return metrics, metrics.copy()


def test_gating_and_early_stop_counts_only_structural_survivors(tmp_path):
    paths, up1 = setup_search(tmp_path)
    esm_calls, structure_calls = [], []

    def esm(candidate):
        _, sequence = read_single_fasta(candidate.iloc[0].fasta_path)
        esm_calls.append(sequence)
        frame = passing_esm(candidate)
        if len(esm_calls) == 1:
            frame[COSINE_METRIC] = .9995
        return frame

    def structure(candidate):
        structure_calls.append(candidate.iloc[0].mutation_spec)
        metrics, _ = passing_structure(candidate)
        metrics['structure_pass'] = len(structure_calls) > 1
        return metrics, metrics.loc[metrics.structure_pass]

    results = run_complex_search(up1, paths, ComplexSearchConfig(target_survivors=2),
                                 .4, esm, structure)
    assert esm_calls == ['RAKR', 'HAKR', 'RAKH', 'HAKH']
    assert len(structure_calls) == 3
    assert len(results['S1_conserved']) == 2
    assert results['T2'].screen_status.tolist() == [
        'ESM2_REJECTED', 'STRUCTURAL_REJECTED', 'CONSERVED', 'CONSERVED',
    ]
    assert results['search_summary']['status'] == 'TARGET_REACHED'
    assert len(pd.read_csv(paths.table('T2_mutation_manifest.csv'))) == 4
    assert len(pd.read_csv(paths.table('ESM2_mutant_comparison.csv'))) == 4
    assert len(pd.read_csv(paths.table('S1_structural_metrics.csv'))) == 4
    assert len(pd.read_csv(paths.table('S1_structurally_conserved.csv'))) == 2


def test_exhaustion_includes_all_cysteine_and_no_other_amino_acids(tmp_path):
    paths, up1 = setup_search(tmp_path)
    result = run_complex_search(up1, paths, ComplexSearchConfig(target_survivors=30),
                                .4, passing_esm, passing_structure)
    assert result['search_summary']['status'] == 'EXHAUSTED'
    assert len(result['S1_conserved']) == 25
    assert result['T2'].iloc[-1].mutation_spec == 'K1C;K4C'


def test_candidate_cap_and_empty_sites_do_not_claim_success(tmp_path):
    paths, up1 = setup_search(tmp_path)
    config = ComplexSearchConfig(max_candidates=2)
    result = run_complex_search(up1, paths, config, .4, passing_esm, passing_structure)
    assert result['search_summary']['status'] == 'CANDIDATE_LIMIT'
    assert len(result['T2']) == 2
    up1['probability'] = .1

    def unexpected(_):
        raise AssertionError('An empty search must not run inference.')

    result = run_complex_search(up1, paths, config, .4, unexpected, unexpected)
    assert result['search_summary']['status'] == 'NO_PROBLEMATIC_SITES'
    for stage in ['T2', 'ESM2', 'S1_metrics', 'S1_conserved']:
        assert result[stage].empty
    assert 'variant_id' in pd.read_csv(paths.table('S1_structurally_conserved.csv')).columns


def test_failure_checkpoints_and_stops_instead_of_exhausting_space(tmp_path):
    paths, up1 = setup_search(tmp_path)

    def fail(_):
        raise RuntimeError('GPU unavailable')

    with pytest.raises(RuntimeError, match='GPU unavailable'):
        run_complex_search(up1, paths, ComplexSearchConfig(), .4, fail, passing_structure)
    summary = json.loads(paths.table('T2_search_summary.json').read_text())
    assert summary['status'] == 'FAILED'
    assert len(pd.read_csv(paths.table('T2_mutation_manifest.csv'))) == 1


def test_standalone_s1_complex_gates_before_structure(tmp_path, monkeypatch):
    workflow = OptimizationWorkflow(tmp_path, WorkflowConfig(mode='sophisticated'))
    workflow.T1('KK')
    manifest = pd.DataFrame([
        dict(variant_id=variant, mutation_spec=spec, mutation_count=2,
             fasta_path=str(tmp_path / f'{variant}.fasta'), status='PASS')
        for variant, spec in [('RR', 'K1R;K2R'), ('HR', 'K1H;K2R')]
    ])
    esm = pd.DataFrame([
        dict(variant_id='RR', esm2_screen_pass=False, esm2_failure_reason='ESM2_COSINE_GATE'),
        dict(variant_id='HR', esm2_screen_pass=True, esm2_failure_reason=''),
    ])
    class Scorer:
        def release(self):
            pass
    workflow._esm2_scorer = Scorer()
    monkeypatch.setattr(workflow, 'ESM2_complex', lambda _: esm)
    calls = []

    def structure(candidate, **kwargs):
        calls.append(candidate.variant_id.tolist())
        return passing_structure(candidate)

    monkeypatch.setattr(workflow, 'S1_basic', structure)
    metrics, conserved = workflow.S1_complex(manifest)
    assert calls == [['HR']]
    assert conserved.variant_id.tolist() == ['HR']
    assert metrics.set_index('variant_id').loc['RR', 'analysis_status'] == 'SKIPPED_ESM2'


def test_empty_structure_stage_does_not_invoke_predictor(tmp_path):
    from therapeutic_optimization.complex_search import MANIFEST_COLUMNS
    from therapeutic_optimization.config import StructuralThresholds
    from therapeutic_optimization.structural_analysis.pipeline import run_s1

    class Predictor:
        def predict_batch(self, *args):
            raise AssertionError('No structure prediction should run for an empty manifest.')

    metrics, conserved = run_s1(pd.DataFrame(columns=MANIFEST_COLUMNS),
                                ProjectPaths.from_root(tmp_path), Predictor(), StructuralThresholds())
    assert metrics.empty and conserved.empty


def test_basic_mode_dispatch_and_separate_table_names(tmp_path):
    workflow = OptimizationWorkflow(tmp_path)
    workflow.T1('KAKK')
    up1 = pd.DataFrame({'lysine_position': [1, 4], 'probability': [.9, .9]})
    basic = workflow.T2(up1)
    assert basic.mutation_spec.tolist() == ['K1A', 'K4A']
    assert workflow.paths.table('T2_mutation_manifest.csv').name == 'T2_basic_mutation_manifest.csv'
    complex_workflow = OptimizationWorkflow(tmp_path, WorkflowConfig(mode='sophisticated'))
    assert complex_workflow.paths.table('T2_mutation_manifest.csv').name == 'T2_complex_mutation_manifest.csv'
    assert workflow.paths.table('T2_mutation_manifest.csv').exists()
    for stage in ['T1', 'UP1', 'T2', 'ESM2', 'S1', 'R1', 'UB2', 'R2']:
        assert callable(getattr(workflow, stage + '_basic'))
        assert callable(getattr(workflow, stage + '_complex'))


def test_notebook_mode_switch_and_python_syntax():
    notebook = json.loads((Path(__file__).parents[1] / 'notebooks/therapeutic_optimization_colab.ipynb').read_text())
    code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    assert code[0].startswith('WORKFLOW_MODE = "sophisticated"')
    for index, source in enumerate(code):
        compile(source, f'notebook cell {index}', 'exec')


@pytest.mark.parametrize('mode', ['basic', 'sophisticated'])
def test_full_workflow_offline_and_no_repeated_complex_inference(tmp_path, monkeypatch, mode):
    import numpy as np
    import therapeutic_optimization.workflow as workflow_module
    from therapeutic_optimization.config import ESM2AnalysisConfig
    from therapeutic_optimization.esm2_analysis import SequenceScore

    config = WorkflowConfig(mode=mode, complex_search=ComplexSearchConfig(target_survivors=2))
    workflow = OptimizationWorkflow(tmp_path, config)

    class Scorer:
        config = ESM2AnalysisConfig(model_name='fake/esm2', save_per_residue=False)
        sequences = []
        releases = 0

        def score_sequence(self, sequence):
            self.sequences.append(sequence)
            embeddings = np.ones((len(sequence), 3))
            return SequenceScore(sequence, np.full(len(sequence), -.5), embeddings,
                                 embeddings.mean(axis=0))

        def release(self):
            self.releases += 1

    class Ubi:
        variants = []
        releases = 0

        def release(self):
            self.releases += 1

        def predict_sequence(self, sequence, protein_id, variant_id, output_dir):
            self.variants.append(variant_id)
            return pd.DataFrame([
                dict(variant_id=variant_id, lysine_position=i, probability=.9 if i != 3 else .1,
                     threshold=.4)
                for i, aa in enumerate(sequence, 1) if aa == 'K'
            ])

    scorer, ubi = Scorer(), Ubi()
    workflow._esm2_scorer = scorer
    workflow._ubi_predictor = ubi
    workflow._structure_predictor = object()
    structure_calls = []

    def structure(manifest, paths, predictor, thresholds, predict_structures=True, **kwargs):
        structure_calls.append((len(manifest), kwargs))
        metrics, conserved = passing_structure(manifest)
        return metrics, conserved

    monkeypatch.setattr(workflow_module, 'run_s1', structure)
    monkeypatch.setattr(workflow_module, 'find_rank1_structure', lambda _: tmp_path / 'WT.pdb')
    result = workflow.run_all('KAKK')
    assert len(result['S1_conserved']) == 2
    assert len(result['R2_all']) == 2
    assert len(ubi.variants) == 3  # WT plus precisely two survivors.
    if mode == 'sophisticated':
        assert result['search_summary']['status'] == 'TARGET_REACHED'
        assert scorer.sequences == ['KAKK', 'RAKR', 'HAKR']  # WT scored only once.
        assert len(structure_calls) == 2
        assert structure_calls[0][1]['wt_structure_path'] is None
        assert structure_calls[1][1]['wt_structure_path'] == tmp_path / 'WT.pdb'
        assert structure_calls[0][1]['batch_output_dir'] != structure_calls[1][1]['batch_output_dir']
        assert ubi.releases == 1
        assert scorer.releases >= 2
        # Display cells must not rerun either expensive screen.
        workflow.ESM2_complex(result['T2'])
        workflow.S1_complex(result['T2'])
        assert len(scorer.sequences) == 3
        assert len(structure_calls) == 2
    else:
        assert 'search_summary' not in result
        assert len(structure_calls) == 1
        assert result['T2'].mutation_count.tolist() == [1, 1]
