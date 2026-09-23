from __future__ import annotations

from pathlib import Path

### import all the pathways

import pandas as pd
from therapeutic_optimization import transformations, unit_operations, config


class Process_flow_diagram():
    """Top-level orchestration with explicit T1/UP1/T2/ESM2/S1/R1/UB2/R2 stages."""

    def __init__(
        self, seq, property, mutant_mode, repeat_with_ESM_peek,
        project_root: str | Path, alt_AA = None 
    ) -> None:
        assert(type(property) == str)
        assert(type(repeat_with_ESM_peek) == bool)
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
        self.mutant_manifest = pd.dataFrame(columns = ["mutant_id", "path", "score", 
                                                          "ESM-2 perp", "status"] ) 
        
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

        ###Obtain fasta from storage
        sequence = self.read_single_fasta(Path.wt_fasta)

        ###Predict sites and return a string of list of the sites
        if (self.property == "ubiquitination" ):
            model = unit_operations.tUP1_EUP.build_predictor()
            muatation_sites = unit_operations.tUP1_EUP.predict_sites(model,)["sites"]
            unit_operations.UO1.assertQC()
        return str(muatation_sites)
                   
    def T2(muatation_sites, self):
        "This will generate all the mutant sequences according to the user mode"
        transformations.T2_mutants.assertQC()
        return transformations.T2_mutants.generate_mutant_seq(self.wt_seq, self.mutant_mode, muatation_sites, self.alt_AA)
     
    def UO2(self) -> pd.DataFrame:
        """Run and return UO2 from the unit_operations file"""
        return 

    def run_all(self, sequence: str, protein_id: str = 'WT',
    ) -> dict[str, object]:
        
        """Execute the complete workflow. GPU-heavy stages still
        fail loudly if dependencies are missing.

        Will return a dictionary containing the results
        """

        T1 = self.T1()
        UO1 = self.UO1(T1)
        T2 = self.T2(UO1)
        UO2 = self.UO2(T2)
        SEP1 = self.SEP1(UO2)
        if (SEP1.recycle != None):
            #optimize to do more amino acids, esm peeking
            pass
        T3 = self.results



        #if ESM is done: 
        #    try:
        #    finally:
                # This model is not needed again in run_all; release its memory
                # before structure prediction and later EUP mutant inference.
        #        self.esm2_scorer.release()

        
        results = {
            'mutation sites': T2,
            'Structural results overview': UO2,
            'Ranked': T3,
            "Optimization repeated": SEP1
        }
        if self._complex_results is not None:
            results['search_summary'] = self._complex_results['search_summary']
        return results

