from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..io import read_single_fasta


@dataclass(frozen=True)
class StructurePrediction:
    variant_id: str
    fasta_path: Path
    output_dir: Path


class ColabFoldPredictor:
    """Thin structure-predictor adapter around the colabfold_batch CLI."""

    name = 'ColabFold'

    def __init__(self, executable: str = 'colabfold_batch', extra_args: tuple[str, ...] = ()) -> None:
        self.executable = executable
        self.extra_args = tuple(extra_args)

    def resolve_executable(self) -> str:
        explicit = Path(self.executable).expanduser()
        if explicit.is_file():
            return str(explicit.resolve())
        resolved = shutil.which(self.executable)
        if resolved is None:
            raise RuntimeError(
                f"Structure predictor {self.executable!r} was not found. Install ColabFold "
                'or provide an explicit executable path.'
            )
        return resolved

    def predict(self, fasta_path: str | Path, output_dir: str | Path) -> Path:
        """Predict one structure (kept for adapter backwards compatibility)."""
        prediction = StructurePrediction('structure', Path(fasta_path), Path(output_dir))
        return self.predict_batch([prediction], Path(output_dir) / 'batch')[prediction.variant_id]

    def predict_batch(
        self,
        predictions: Iterable[StructurePrediction],
        batch_output_dir: str | Path,
    ) -> dict[str, Path]:
        """Run all requested sequences through a single ColabFold invocation."""
        predictions = list(predictions)
        if not predictions:
            return {}

        variant_ids = [prediction.variant_id for prediction in predictions]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError('Structure prediction variant IDs must be unique within a batch.')

        batch_output_dir = Path(batch_output_dir).resolve()
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        executable = self.resolve_executable()

        # Stable, simple query names avoid ColabFold filename sanitization while
        # preserving the caller's variant IDs in the returned mapping.
        query_names: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix='colabfold-input-') as temporary_dir:
            batch_fasta = Path(temporary_dir) / 'batch.fasta'
            with batch_fasta.open('w', encoding='utf-8') as handle:
                for index, prediction in enumerate(predictions):
                    fasta_path = Path(prediction.fasta_path).resolve()
                    if not fasta_path.exists():
                        raise FileNotFoundError(fasta_path)
                    _header, sequence = read_single_fasta(fasta_path)
                    query_name = f'query_{index:06d}'
                    query_names[prediction.variant_id] = query_name
                    handle.write(f'>{query_name}\n{sequence}\n')

            command = [executable, str(batch_fasta), str(batch_output_dir), *self.extra_args]
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        log_path = batch_output_dir / 'colabfold_run.log'
        log_path.write_text(result.stdout, encoding='utf-8')
        if result.returncode != 0:
            raise RuntimeError(
                f'ColabFold batch failed for {len(predictions)} sequences with code '
                f'{result.returncode}. See {log_path}.'
            )

        structures: dict[str, Path] = {}
        for prediction in predictions:
            query_name = query_names[prediction.variant_id]
            source = find_rank1_structure(batch_output_dir, query_name=query_name)
            destination_dir = Path(prediction.output_dir).resolve()
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f'rank_001{source.suffix.lower()}'
            shutil.copy2(source, destination)
            structures[prediction.variant_id] = destination
        return structures

def find_rank1_structure(output_dir: str | Path, query_name: str | None = None) -> Path:
    """Locate the highest-ranked PDB/CIF emitted by ColabFold."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)

    files = [
        path
        for path in output_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in {'.pdb', '.cif', '.mmcif'}
        and (query_name is None or path.name.startswith(f'{query_name}_'))
    ]
    if not files:
        query_detail = f' for query {query_name!r}' if query_name is not None else ''
        raise FileNotFoundError(f'No PDB/CIF structure found{query_detail} under {output_dir}.')

    def priority(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if 'rank_001' in name or 'rank_1' in name:
            return (0, name)
        if 'ranked_0' in name:
            return (1, name)
        if 'unrelaxed' in name and 'rank' in name:
            return (2, name)
        if 'relaxed' in name and 'rank' in name:
            return (3, name)
        return (9, name)

    return sorted(files, key=priority)[0]
