from pathlib import Path
import subprocess
from uuid import uuid4

import shutil
#move structural analysis into here
import sys
sys.path.insert(0, "/content")  # Replace with your .py file's folder

from colabfold_runner import fold_one, fold_batch, fold
from Bio import SeqIO


#look over class and how it compares to standard code surronding colabfold/alphafold
class ColabFoldPredictor:
    """Thin structure-predictor adapter around the colabfold_batch CLI."""

    name = 'ColabFold'

    def __init__(self, executable: str = 'colabfold_batch',) -> None:
        self.executable = executable

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

    def predict_one(self, wt_sequence, protein_id,):
        best_pdb = fold_one(sequence=wt_sequence, name=protein_id, output_dir="/content/fold_wt",
        num_models=5, num_recycles=3,)
        return(best_pdb)

    def predict_batch(self, batch: list, portion: tuple, batch_output_dir: Path,) -> dict[str, list[Path]]:
        """Takes a list of seqs and queries colab fold."""
        predictions = list(batch)
        if not predictions:
            return {}
        variant_ids = [prediction.variant_id for prediction in predictions]
        
        batch_output_dir = Path(batch_output_dir).resolve()
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        executable = self.resolve_executable()
        indices = str(portion[0]) + "-" + str(portion[1])
        # Stable, simple query names avoid ColabFold filename sanitization while
        # preserving the caller's variant IDs in the returned mapping.
        query_names: dict[str, str] = {}
        batch_fastas_dir = Path.cwd() / 'storage' / 'batch_fastas'
        batch_fastas_dir.mkdir(parents=True, exist_ok=True)
        batch_fasta = batch_fastas_dir / f'batch_{indices}.fasta'
        with batch_fasta.open('w', encoding='utf-8') as handle:
            for index, prediction in enumerate(predictions):
                fasta_path = Path(prediction.fasta_path).resolve()

                if not fasta_path.exists():
                    raise FileNotFoundError(fasta_path)

                record = next(SeqIO.parse(fasta_path, 'fasta'))
                sequence = str(record.seq)

                query_name = f'query_{index:06d}'
                query_names[prediction.variant_id] = query_name
                handle.write(f'>{query_name}\n{sequence}\n')


        command = [executable, str(batch_fasta), str(batch_output_dir)]
        result = subprocess.run(command,text=True,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,check=False,)

    
        log_path = batch_output_dir / 'colabfold_run.log'
        log_path.write_text(result.stdout, encoding='utf-8')
        if result.returncode != 0:
            raise RuntimeError(
                f'ColabFold batch failed for {len(predictions)} sequences with code '
                f'{result.returncode}. See {log_path}.'
                    )
        
        structures: dict[str, list[Path]] = {}
        for prediction in predictions:
            query_name = query_names[prediction.variant_id]
            destination_dir = Path(prediction.output_dir).resolve()
            destination_dir.mkdir(parents=True, exist_ok=True)

            source_files = sorted(
                source
                for source in batch_output_dir.rglob(f'{query_name}*')
                if source.is_file() and source.suffix.lower() in {'.pdb', '.cif', '.mmcif'}
            )
            if not source_files:
                raise FileNotFoundError(
                    f'No structures found for {prediction.variant_id} '
                    f'with query name {query_name} in {batch_output_dir}.'
                )

            variant_structures: list[Path] = []
            for source in source_files:
                destination = destination_dir / source.name
                shutil.copy2(source, destination)
                variant_structures.append(destination)
            structures[prediction.variant_id] = variant_structures
        return structures
        
#metrics