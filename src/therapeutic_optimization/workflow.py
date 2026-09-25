from __future__ import annotations

from pathlib import Path

from .predictors.tUP1_EUP import EUPPredictor
### import all the pathways

import pandas as pd
from therapeutic_optimization import transformations, predictors


class ProcessFlowDiagram():
    """Top-level orchestration with explicit T1/UP1/T2/ESM2/S1/R1/UB2/R2 stages."""

    def __init__(
        self, seq, property, mutant_mode, 
        project_root: str | Path, alt_AA = None ) -> None:
        assert(type(property) == str)
        self.wt_seq = seq
        self.alt_AA = alt_AA
        #sets the property for the initial insight 
        self.property = property
        #sets the algorithim for generating the mutants, later this will be turned and the
        #tool chain will just recycle into more complex combinations and less expensive 
        #searches until it finds good hits 
        self.mutant_mode =  mutant_mode

        #TODO: Understand the configuration threading. it seems over complicated? my 
        #intuition tells me to keep it as simple as possible, threading arguments to 
        #definitons securely ......
        self.PredictorConfiguration = None
        ##intialize data storage

        self.make_drive_folders()
        self.make_local_folders()
        self.mutant_manifest = pd.DataFrame(columns = ["mutant_id", "path", "score", 
                                                          "ESM-2 perp", "status"] )


    def make_drive_folders(self, drive_root: str | Path) -> dict[str, Path]:
        """Create persistent result folders after Google Drive is mounted."""
        drive_root = Path(drive_root).expanduser()

        mounted_drive = Path("/content/drive/MyDrive")
        if not mounted_drive.exists():
            raise RuntimeError(
            "Google Drive is not mounted. Call drive.mount('/content/drive') first."
        )

        drive_root = drive_root.resolve()
        if drive_root != mounted_drive and mounted_drive not in drive_root.parents:
            raise ValueError("drive_root must be located under /content/drive/MyDrive.")

        relative_paths = {
            "results": "results",
            "structures": "structures",
            "run report": "logs",
        }

        self.drive_paths = {name: drive_root / relative_path
        for name, relative_path in relative_paths.items()}

        for path in self.drive_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        return self.drive_paths

    def make_local_folders(self) -> dict[str, Path]:
        """Create the fast, temporary workspace used during computation."""
        relative_paths = {
            "wt_fasta": "wt_fasta",
            "mut_fasta" : "mut_fasta",
            "wt_structures": "structures/wt",
            "mutant_structures": "structures/mutants",
            "figures": "figures",
            "logs": "logs",
        }

        self.local_paths = {name: self.local_root / relative_path
        for name, relative_path in relative_paths.items()
        }

        for path in self.local_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        return self.local_paths

    def prepareinput(self):
        pass

        
    def T1(self):
        """ Transforms the user input into the required format for the predictor.
        This will also save a fasta copy of the wildtype sequence to the fastas 
        folder in storage. 
        """
        if (self.property == "ubiquitination" ):
            return transformations.T1_input.prepare_wt_input(self.wt_seq)

    def UO1(self):
        """This transforms the reformatted users input into which sites should 
        be mutated"""

        sequence = self.wt_seq
        ###Predict sites and return a string of list of the sites
        if (self.property == "ubiquitination" ):
            model = EUPPredictor()
            model.build_predictor()
            mutation_sites = model.predict_sites(sequence)            
            model.assertQC()
            return mutation_sites.loc[mutation_sites["is_positive"], "site"].tolist()

    def T2(self, mutation_sites):
        """This will generate all the mutant sequences according to the user mode and return a string 
        of the sequences """
        transformations.T2_mutants.assertQC()
        return transformations.T2_mutants.generate_mutant_seq(self.wt_seq, self.mutant_mode, mutation_sites, self.alt_AA)
     
    def UO2(self) -> pd.DataFrame:
        """Run and return UO2 from the predictors file"""
        return 

    def run_all(self, sequence: str, protein_id: str = 'WT',
    ) -> dict[str, object]:
        
        """Execute the complete workflow. GPU-heavy stages still
        fail loudly if dependencies are missing.

        Will return a dictionary containing the results
        """

        self.make_folders()

        #prepare package, install dependencies, stow wt info
        self.T1()

        #predict mutation sites
        self.UO1()

        #generate mutant structures
        T2 = self.T2()
        UO2 = self.UO2(T2)
        return UO2

        #if ESM is done: 
        #    try:
        #    finally:
                # This model is not needed again in run_all; release its memory
                # before structure prediction and later EUP mutant inference.
        #        self.esm2_scorer.release()

