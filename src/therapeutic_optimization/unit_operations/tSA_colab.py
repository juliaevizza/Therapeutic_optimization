from pathlib import Path
import subprocess
from uuid import uuid4
import shutil
import sys
from abc import ABC, abstractmethod
from .UO2 import StructuralAnalysis

from colabfold_runner import fold_one, fold_batch, fold
from Bio import SeqIO

import numpy as np
import pandas as pd
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile


from ..config import ProjectPaths

#TODO include wt type in batch 

#look over class and how it compares to standard code surronding colabfold/alphafold
class ColabFoldPredictor(StructuralAnalysis):
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

    def predict_structure(self, batch: list, all_batch_output_dir: Path,) -> dict[str, list[Path]]:
        batch_size = 20
        all_structures: dict[str, list[Path]] = {}

        for start in range(0, len(batch), batch_size):
            end = min(start + batch_size, len(batch))
            batch_output_dir = (
                Path(all_batch_output_dir) / f'batch_{start}-{end - 1}')

            batch_structures = self.predict_batch(
                batch=batch[start:end],
                portion_of_batch=(start, end - 1),
                batch_output_dir=batch_output_dir,)
            all_structures.update(batch_structures)
        return all_structures

    def predict_batch(self, batch: list, portion_of_batch: tuple, batch_output_dir: Path,) -> dict[str, list[Path]]:
        """Takes a list of seqs and queries colab fold."""
        predictions = list(batch)
        if not predictions:
            return {}
        variant_ids = [prediction.variant_id for prediction in predictions]
        
        batch_output_dir = Path(batch_output_dir).resolve()
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        executable = self.resolve_executable()
        indices = str(portion_of_batch[0]) + "-" + str(portion_of_batch[1])
        # Stable, simple query names avoid ColabFold filename sanitization while
        # preserving the caller's variant IDs in the returned mapping.
        query_names: dict[str, str] = {}
        batch_fastas_dir = ProjectPaths.from_root(Path.cwd()).batch_fastas
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

    def standardize_analysis(self, structures: dict[str, list[Path]],) -> dict[str, object]:        
        """This will standardize the output as predicted by the colabfold.
        It will compare the appropriate metrics from the predictor to score
        the sequence based on structural conservation. All the metrics will be
        mean pooled into a metric that will be compared with a threshold"""
        mutant_results: dict[str, dict[str, object]] = {}

        for variant_id, structure_paths in structures.items():
        # Least-squares alignment using every selected Cα atom.
        # No outlier removal.
            wt = structures["Wildtype"]
            mut_aligned, transformation = struc.superimpose(wt, variant_id)
            rmsd = float(struc.rmsd(wt, mut_aligned))
            displacement = np.linalg.norm(wt.coord - mut_aligned.coord, axis=1)
            global_lddt = float(struc.lddt(wt, variant_id))
            residue_lddt = struc.lddt(wt, variant_id, aggregation="residue")
            structure_results = []


            # TODO: calculate metrics by comparing this structure
            # against the WT/reference structure.
            for structure_path in structure_paths:
                raw_metrics = { "RMSD" = displacement; "global_lddt" = global_lddt; "residue_lddt" = residue_lddt;}
                    #Per-residue lDDT
                    #Local effective strain
                    #predicted stability 
                    #'rmsd': 
                    # 'confidence': ...,
                    

            # TODO: scale every metric to [0, 1].
            # Metrics where lower is better must be inverted here.
                scaled_metrics = {
                # 'rmsd': ...,
                # 'contact_change': ...,
                # 'local_displacement': ...,
                }

            # TODO: choose metric weights.
                metric_score = sum(scaled_metrics.values()) / len(scaled_metrics)

            # TODO: extract confidence from the structure output.
            # For PDB files, ColabFold commonly stores pLDDT
            # in the B-factor column.
                confidence_score = float(raw_metrics.get('confidence', 1.0))

                structure_results.append({
                    'metric_score': metric_score,
                    'confidence_score': confidence_score,
                })

            confidence_total = sum(
                float(result['confidence_score'])
                for result in structure_results
            )

            if confidence_total > 0:
                mutant_score = sum(
                    float(result['metric_score'])
                    * float(result['confidence_score'])
                    for result in structure_results
                ) / confidence_total
            else:
                mutant_score = sum(
                    float(result['metric_score'])
                    for result in structure_results
                ) / len(structure_results)

            mutant_confidence = sum(
                float(result['confidence_score'])
                for result in structure_results
            ) / len(structure_results)

            mutant_results[variant_id] = {
                'structures': structure_results,
                'structural_score': mutant_score,
                'confidence_score': mutant_confidence,
            }

        confidence_total = sum(
            float(result['confidence_score'])
            for result in mutant_results.values()
        )

        overall_score = sum(
            float(result['structural_score'])
            * float(result['confidence_score'])
                for result in mutant_results.values()
            ) / confidence_total

        return {
            'variants': mutant_results,
            'overall_structural_score': overall_score,
        }
