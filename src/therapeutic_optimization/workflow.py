from __future__ import annotations

from pathlib import Path

from .predictors.tUP1_EUP import EUPPredictor
from .config import WorkflowConfig
### import all the pathways

import pandas as pd
from .transformations.T1_input import prepare_wt_input
from .transformations.T2_mutants import (
    assertQC as assert_qcT2,
    generate_mutant_seq,
)

class ProcessFlowDiagram():
    """Top-level orchestration with explicit T1/UP1/T2/ESM2/S1/R1/UB2/R2 stages."""

    def __init__(self, seq: str, property: str, mutant_mode: str, project_root: str | Path,
    alt_AA=None, protein_id: str = "WT", config: WorkflowConfig | None = None,
    drive_root: str | Path | None = None,) -> None:
        assert(type(property) == str)
        self.wt_seq = seq
        self.alt_AA = alt_AA
        self.normalize_alt_AA()
        self.protein_id = protein_id
        self.config = config or WorkflowConfig()

        #intialize project root
        self.project_root = Path(project_root).expanduser().resolve()
        self.local_root = self.project_root

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

        self.make_local_folders()

        self.drive_paths: dict[str, Path] = {}
        if drive_root is not None:
            self.make_drive_folders(drive_root)

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

    def normalize_alt_AA(self):
        alt_AA = self.alt_AA
        if alt_AA is None:
            self.alt_AA = ("R",)
        elif isinstance(alt_AA, str):
            self.alt_AA = (alt_AA.upper(),)
        else:
            self.alt_AA = tuple(aa.upper() for aa in alt_AA)


        
    def T1(self):
        """ Transforms the user input into the required format for the predictor.
        This will also save a fasta copy of the wildtype sequence to the fastas 
        folder in storage. 

    return transformations.T1_input.prepare_wt_input(
    )
        """
        if (self.property == "ubiquitination" ):
            return prepare_wt_input(
        sequence=self.wt_seq,
        protein_id=self.protein_id,
        input_directory=self.local_paths["wt_fasta"],)

    def UO1(self):
        """This transforms the reformatted users input into which sites should 
        be mutated"""

        sequence = self.wt_seq
        ###Predict sites and return a string of list of the sites
        if (self.property == "ubiquitination" ):
            model = EUPPredictor(threshold=self.config.ubiquitination.threshold,
                        eup_repo_dir=self.config.ubiquitination.eup_repo_dir,
                        model_cache_dir=self.config.ubiquitination.model_cache_dir,)
            model.build_predictor()
            prediction_results= model.predict_sites(sequence)            
            model.assert_qc(prediction_results)
            return prediction_results.loc[prediction_results["is_positive"], "site"].tolist()

    def T2(self, mutation_sites):
        """This will generate all the mutant sequences according to the user mode and return a string 
        of the sequences """
        assert_qcT2()
        return generate_mutant_seq(self.wt_seq, self.mutant_mode, mutation_sites, self.alt_AA,)
    def UO2(self) -> pd.DataFrame:
        """Run and return UO2 from the predictors file"""

        return 
    
    def run_all(self) -> dict[str, object]: 
        
        """Execute the complete workflow. GPU-heavy stages 
        fail loudly if dependencies are missing.

        Will return a dictionary containing the results
        """


        #prepare package, install dependencies, stow wt info
        t1_result = self.T1()

        #predict mutation sites
        mutation_sites = self.UO1()

        #generate mutant structures
        T2 = self.T2(mutation_sites)


        return {
        "T1": t1_result,
        "UO1": mutation_sites,
        "T2": T2,
        }        #run strutctural prediction on mutant structures
        #UO2 = self.UO2(T2)
        #sreturn UO2
    
