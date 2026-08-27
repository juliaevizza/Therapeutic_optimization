from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


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
        fasta_path = Path(fasta_path).resolve()
        output_dir = Path(output_dir).resolve()
        if not fasta_path.exists():
            raise FileNotFoundError(fasta_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        executable = self.resolve_executable()
        command = [executable, str(fasta_path), str(output_dir), *self.extra_args]
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (output_dir / 'colabfold_run.log').write_text(result.stdout, encoding='utf-8')
        if result.returncode != 0:
            raise RuntimeError(
                f'ColabFold failed for {fasta_path.name} with code {result.returncode}. '
                f'See {output_dir / "colabfold_run.log"}.'
            )
        return find_rank1_structure(output_dir)


def find_rank1_structure(output_dir: str | Path) -> Path:
    """Locate the highest-ranked PDB/CIF emitted by ColabFold."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)

    files = [
        path
        for path in output_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in {'.pdb', '.cif', '.mmcif'}
    ]
    if not files:
        raise FileNotFoundError(f'No PDB/CIF structure found under {output_dir}.')

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
