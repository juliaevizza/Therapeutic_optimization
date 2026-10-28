#This file has been reviewed and is completely set 
from pathlib import Path
import subprocess
import shutil
import pandas as pd
import numpy as np
import os

from .structural import StructuralAnalysis

from Bio import SeqIO
from Bio.PDB import PDBParser, Superimposer, NeighborSearch

#need to import PDB parser

#look over class and how it compares to standard code surronding colabfold/alphafold
class ColabFoldPredictor(StructuralAnalysis):
    """
    Thin structure-predictor adapter around the colabfold_batch CLI.
    """

    name = 'ColabFold'

    def __init__(self, executable: str = 'colabfold_batch',) -> None:
        """
        TODO
        """
        self.executable = executable

    def resolve_executable(self) -> str:
        """
        TODO
        """
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

    def predict_structures(self, mutant_manifest: pd.DataFrame, all_batch_output_dir: Path,batch_size: int = 20,) -> dict[str, list[Path]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        assert isinstance(mutant_manifest, pd.DataFrame), "mutant_manifest must be a pandas DataFrame."

        assert {"mutant_id", "sequence"}.issubset(mutant_manifest.columns), "Manifest must contain mutant_id and sequence columns."

        assert not mutant_manifest.empty, "No sequences were supplied."
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer."

        ids = mutant_manifest["mutant_id"]

        assert ids.map( lambda value: isinstance(value, str) and bool(value.strip())).all(), "Every protein must have a nonempty string ID."

        assert ids.is_unique, "Protein IDs must be unique."
        assert ids.eq("WT").sum() == 1, "Exactly one WT reference is required."

        assert mutant_manifest["sequence"].map( lambda value: isinstance(value, str) and bool(value.strip())).all(), "Every protein must have a nonempty sequence."

        executable = self.resolve_executable()
        output_root = Path(all_batch_output_dir)
        structures = {}
        environment = dict(os.environ, MPLBACKEND="Agg")

        for batch_number, start in enumerate(
        range(0, len(mutant_manifest), batch_size)):
            batch = mutant_manifest.iloc[start:start + batch_size]
            batch_dir = output_root / f"batch_{batch_number:03d}"
            batch_dir.mkdir(parents=True, exist_ok=False)

            batch_csv = batch_dir / "input.csv"
            batch[["mutant_id", "sequence"]].rename(columns={"mutant_id": "id"}).to_csv(batch_csv, index=False)

            command = [executable, str(batch_csv), str(batch_dir), "--num-models", "5","--num-seeds", "1",]

            log_path = batch_dir / "colabfold_run.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env=environment,
            )

            if result.returncode != 0:
                raise RuntimeError(f"ColabFold failed. See {log_path}")

            for mutant_id in batch["mutant_id"]:

                model_paths = sorted(
                batch_dir.glob(f"{mutant_id}_unrelaxed_rank_*.pdb") )

                assert model_paths, (f"No structures were produced for {mutant_id}. " f"Inspect {log_path} for the ColabFold error.")
                assert all(path.stat().st_size > 0 for path in model_paths), (f"An empty PDB file was produced for {mutant_id}. "f"Inspect {log_path}.")
                assert len(model_paths) == 5, (f"Expected 5 structures for {mutant_id}, "f"but found {len(model_paths)}. Inspect {log_path}.")
                
                structures[mutant_id] = model_paths
        return structures


    def run_metrics(self, manifest: pd.DataFrame, structures: dict[str, list[Path]])-> None:
            """
            This will run the metrics on the PDB file to determine the
            stuctural differences between the wt and the mutant and write the score
            to the mutant manifest for info tracking purposes.
            """
            parser = PDBParser(QUIET=True)

            # Generate WT once and read its reference atoms once.
            wt_models = sorted(structures.get("WT", []))

            if not wt_models:
                raise RuntimeError( "Cannot calculate structural scores: no WT reference structure was produced. ")

            wt_path = wt_models[0]
            
            wt_atoms = self.read_ca_atoms(wt_path)
            for mutant_id in manifest["mutant_id"]:
                if mutant_id == "WT":
                    continue
                mutants_pdb_paths = structures.get(mutant_id, [])
                mut_atoms = [self.read_ca_atoms(pdb_path) for pdb_path in mutants_pdb_paths]
                manifest.loc[manifest["mutant_id"] == mutant_id, "score"] = self.score_mutant(wt_atoms, mut_atoms)

    
    #########Helpers for metrics#########    
    
    def score_mutant(self, wt_atoms, mut_atoms: list) -> float:
        """Score a mutant using all of its predicted models. Pass into this as
        processed atoms. This function returns the float corresponding to the mutants
        score.
        """
        if not mut_atoms:
            raise ValueError("At least one mutant structure is required.")

        rmsd_values = []
        global_lddt_values = []

        for model_atoms in mut_atoms:
            # This helper also checks matching chain/residue IDs and order.
            global_lddt = self.global_ca_lddt(wt_atoms, model_atoms)
            rmsd = self.rmsd(wt_atoms, model_atoms)

            if not np.isfinite([rmsd, global_lddt]).all():
                raise ValueError("A mutant model produced an undefined metric.")

            rmsd_values.append(rmsd)
            global_lddt_values.append(global_lddt)

        mean_rmsd = float(np.mean(rmsd_values))
        mean_global_lddt = float(np.mean(global_lddt_values))

        # TODO: Replace this placeholder with the actual scoring function.
        score = 100 * mean_global_lddt
        return score

    @staticmethod
    def global_ca_lddt(wt_atoms, mutant_atoms, cutoff=15.0):
        """
        Return one global Cα-lDDT score.
        """
        wt_ids = [a.get_full_id()[2:4] for a in wt_atoms]
        mutant_ids = [a.get_full_id()[2:4] for a in mutant_atoms]
    
        if wt_ids != mutant_ids:
            raise ValueError("Cα atoms must have matching chain/residue IDs and order.")
    
        positions = {atom: j for j, atom in enumerate(wt_atoms)}
        total_score = 0.0
        pair_count = 0
    
        for a, b in NeighborSearch(wt_atoms).search_all(cutoff):
            reference_distance = float(a - b)
    
            if reference_distance >= cutoff:
                continue
    
            j, k = positions[a], positions[b]
            mutant_distance = float(mutant_atoms[j] - mutant_atoms[k])
            difference = abs(reference_distance - mutant_distance)
    
            total_score += sum(
                difference < t for t in (0.5, 1.0, 2.0, 4.0)
            ) / 4
            pair_count += 1
    
        return total_score / pair_count if pair_count else float("nan")

    #TODO double check
    @staticmethod
    def rmsd(wt_atoms, mutant_atoms):
        fit = Superimposer()
        fit.set_atoms(wt_atoms, mutant_atoms)
        return float(fit.rms)

    @staticmethod
    def read_ca_atoms(pdb_path):
        pdb_path = Path(pdb_path)
        if not pdb_path.is_file():
            raise FileNotFoundError(pdb_path)

        model = PDBParser(QUIET=True).get_structure(
            pdb_path.stem, str(pdb_path)
        )[0]

        atoms = {(chain.id, residue.id): residue["CA"]
            for chain in model
            for residue in chain
            if residue.id[0] == " " and "CA" in residue}

        # Use consistent chain/residue ordering for every structure.
        ordered_atoms = [atoms[key] for key in sorted(atoms)]

        if len(ordered_atoms) < 3:
            raise ValueError(f"Need at least three Cα atoms: {pdb_path}")
        if not np.isfinite([atom.coord for atom in ordered_atoms]).all():
            raise ValueError(f"Non-finite coordinates: {pdb_path}")
        return ordered_atoms
