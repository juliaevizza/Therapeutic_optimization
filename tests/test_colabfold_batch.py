from __future__ import annotations

from pathlib import Path

from therapeutic_optimization.structural_analysis.colabfold import (
    ColabFoldPredictor,
    StructurePrediction,
)


def test_predict_batch_uses_one_command_and_preserves_variant_layout(tmp_path, monkeypatch):
    fasta_a = tmp_path / 'a.fasta'
    fasta_b = tmp_path / 'b.fasta'
    fasta_a.write_text('>original-a\nAAAA\n', encoding='utf-8')
    fasta_b.write_text('>original-b\nAKAA\n', encoding='utf-8')
    calls = []

    class Result:
        returncode = 0
        stdout = 'batch complete'

    def fake_run(command, **kwargs):
        calls.append(command)
        batch_fasta = Path(command[1])
        assert batch_fasta.read_text(encoding='utf-8') == (
            '>query_000000\nAAAA\n>query_000001\nAKAA\n'
        )
        output_dir = Path(command[2])
        (output_dir / 'query_000000_unrelaxed_rank_001_model.pdb').write_text(
            'WT structure', encoding='utf-8'
        )
        (output_dir / 'query_000001_unrelaxed_rank_001_model.pdb').write_text(
            'mutant structure', encoding='utf-8'
        )
        return Result()

    monkeypatch.setattr('shutil.which', lambda executable: f'/usr/bin/{executable}')
    monkeypatch.setattr('subprocess.run', fake_run)
    predictor = ColabFoldPredictor()
    result = predictor.predict_batch(
        [
            StructurePrediction('WT', fasta_a, tmp_path / 'structures' / 'wt'),
            StructurePrediction('K2R', fasta_b, tmp_path / 'structures' / 'mutants' / 'K2R'),
        ],
        tmp_path / 'structures' / 'batch',
    )

    assert len(calls) == 1
    assert result['WT'].read_text(encoding='utf-8') == 'WT structure'
    assert result['K2R'].read_text(encoding='utf-8') == 'mutant structure'
    assert result['WT'] == tmp_path / 'structures' / 'wt' / 'rank_001.pdb'
    assert result['K2R'] == tmp_path / 'structures' / 'mutants' / 'K2R' / 'rank_001.pdb'
    assert (tmp_path / 'structures' / 'batch' / 'colabfold_run.log').read_text(
        encoding='utf-8'
    ) == 'batch complete'
