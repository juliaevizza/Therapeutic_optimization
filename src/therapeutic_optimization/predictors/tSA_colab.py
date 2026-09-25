#This file has been reviewed and is completely set 
from pathlib import Path
import subprocess
import shutil
from .Predictor2_struct import StructuralAnalysis

from colabfold_runner import fold_one, fold_batch, fold
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

    def predict_batch(self, batch: list, portion_of_batch: tuple, batch_output_dir: Path,) -> dict[str, list[Path]]:
        """
        Takes a list of seqs and queries colab fold with a portion of them.
        """
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

        #TODO
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

        ##runs batches through a command line call
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

