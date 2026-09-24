from pathlib import Path
import subprocess
from uuid import uuid4
from Bio.PDB import PDBParser, Superimposer, NeighborSearch

import shutil
#move structural analysis into here
import sys
sys.path.insert(0, "/content")  # Replace with your .py file's folder

from abc import ABC, abstractmethod
from colabfold_runner import fold_one, fold_batch, fold
from Bio import SeqIO


#look over class and how it compares to standard code surronding colabfold/alphafold
class StructuralAnalysis(ABC):
    """Thin structure-predictor adapter around whichever model I am calling"""

    @abstractmethod
    def __init__() -> None:
        pass

    @abstractmethod
    def predict_structure(self, batch: list, portion: tuple, batch_output_dir: Path,) -> dict[str, list[Path]]:
        """Takes a list of seqs in list "batch", queries the structural predictor, and ."""

    @abstractmethod
    def standardize_analysis():
        """This will be what ensures that output is always consistent leaving the structural analysis. """


## def maybe also an ESM_metrics? 
#metrics
#metrics will be run on PDB files and PDB FILES ONLY! (most common format output)
    def run_PDB_metrics(self, manifest: pd.DataFrame, wt_struct_path: Path,):
        # prepare wt pdb
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
    """Map PDB residue numbers to their Cα atoms in the first model."""
    chain = structure[0]
    return {
        residue.id[1]: residue["CA"]
        for residue in chain
        if residue.id[0] == " " and "CA" in residue
    }


### TODO: Need to lock down on these

def global_ca_lddt(wt_atoms, mutant_atoms, cutoff=15.0):
    """Return one global Cα-lDDT score."""
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
    """Return one Cα-lDDT score per residue, in wt_atoms order."""
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