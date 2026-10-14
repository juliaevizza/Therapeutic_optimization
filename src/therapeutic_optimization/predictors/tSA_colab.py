#This file has been reviewed and is completely set 
from pathlib import Path
import subprocess
import shutil
import pandas as pd
from .Predictor2_struct import StructuralAnalysis

from Bio import SeqIO
from Bio.PDB import PDBParser, Superimposer, NeighborSearch



#need to import PDB parser

#TODO include wt type in batch 

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

    def predict_structures(self, batch: list, all_batch_output_dir: Path,) -> dict[str, list[Path]]:
        """
        This will split the mutants into batches of 20 to send to colab via a
        command line call. 
        """
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

    def predict_structures(self, mutant_manifest: pd.DataFrame, all_batch_output_dir: Path,batch_size: int = 20,) -> dict[str, list[Path]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if mutant_manifest.empty:
            return {}

        executable = self.resolve_executable()
        output_root = Path(all_batch_output_dir)
        structures = {}

        for batch_number, start in enumerate(
        range(0, len(mutant_manifest), batch_size)
    ):
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
            )

            if result.returncode != 0:
                raise RuntimeError(f"ColabFold failed. See {log_path}")

            for mutant_id in batch["mutant_id"]:
                model_paths = sorted(
                batch_dir.glob(f"{mutant_id}_unrelaxed_rank_*.pdb") )
                structures[mutant_id] = model_paths

        return structures


    def run_metrics(self, manifest: pd.DataFrame, wt_struct_path: Path,):
            """
            This will run the metrics on the PDB file to determine the
            stuctural differences between the wt and the mutant.
            """
            parser = PDBParser(QUIET=True)
            structures_root = Path("/path/to/storage/structures/mutants")
            wt_structure = parser.get_structure("WT", str(wt_struct_path))
            wt_atoms = ca_atoms_by_position(wt_structure)
    
            for i in manifest.mutant_id:
                #go into folder with corresponding mutant id
                pdb_dir = structures_root / str(i)
                mut_rmsd = []
                mut_global_lddt = []
                mut_residue_lddt = []
                for pdb_path in sorted(pdb_dir.glob("*.pdb")):
                    mut_structure = parser.get_structure(pdb_path.stem, pdb_path)
                    mutant_atoms = ca_atoms_by_position(mut_structure)
    
                    # Biopython fits the mutant atoms onto the WT atoms.
                    fit = Superimposer()
                    fit.set_atoms(wt_atoms, mutant_atoms)
                    mut_rmsd.append(fit.rms)
                    mut_global_lddt.append(global_ca_lddt(wt_atoms, mutant_atoms))
                    mut_residue_lddt.append(residue_ca_lddt(wt_atoms, mutant_atoms))
    
                #perform all metrics compared to wild type
                #mean them
                mut_rmsd_ave = sum(mut_rmsd) / len(mut_rmsd)
                mut_global_lddt_ave = sum(mut_global_lddt) / len(mut_rmsd)
                mut_residue_lddt_ave = sum(mut_residue_lddt) / len(mut_residue_lddt)
                #return overall score for mutant 
                # weight and scale 
                mut_overall_score = mut_residue_lddt_ave + mut_global_lddt_ave + mut_residue_lddt_ave
                manifest.loc[manifest["mutant_id"] == i, "score"] = mut_overall_score
    #Helpers for metrics 
    
    #maybe go and refine implementation to match chains one by one (peptide : enzyme)
    def ca_atoms_by_position(structure,):
        """
        Map PDB residue numbers to their Cα atoms in the first model.
        """
        chain = structure[0]
        return {
            residue.id[1]: residue["CA"]
            for residue in chain
            if residue.id[0] == " " and "CA" in residue
        }
    
    
    ### TODO: Need to lock down on these
    
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
    
    
    def residue_ca_lddt(wt_atoms, mutant_atoms, cutoff=15.0):
        """
        Return one Cα-lDDT score per residue, in wt_atoms order.
        """
        wt_ids = [a.get_full_id()[2:4] for a in wt_atoms]
        mutant_ids = [a.get_full_id()[2:4] for a in mutant_atoms]
    
        if wt_ids != mutant_ids:
            raise ValueError("Cα atoms must have matching chain/residue IDs and order.")
        if len(wt_atoms) < 2:
            raise ValueError("Need at least two Cα atoms.")
    
        positions = {atom: j for j, atom in enumerate(wt_atoms)}
        totals = [0.0] * len(wt_atoms)
        counts = [0] * len(wt_atoms)
    
        for a, b in NeighborSearch(wt_atoms).search_all(cutoff):
            reference_distance = float(a - b)
    
            if reference_distance >= cutoff:
                continue
    
            j, k = positions[a], positions[b]
            mutant_distance = float(mutant_atoms[j] - mutant_atoms[k])
            difference = abs(reference_distance - mutant_distance)
    
            score = sum(
                difference < t for t in (0.5, 1.0, 2.0, 4.0)
            ) / 4
    
            for index in (j, k):
                totals[index] += score
                counts[index] += 1
    
        return [
            total / count if count else float("nan")
            for total, count in zip(totals, counts)
        ]

