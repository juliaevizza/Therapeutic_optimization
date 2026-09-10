from __future__ import annotations

from itertools import combinations, product
from math import prod
from pathlib import Path

import pandas as pd

from ..config import MutationConfig, ProjectPaths


#TODO generate all experimentally interesting data structures of mutants 
"""This defintion encompasses all of the possible modes of generate new mutants based on the models 
predicted sites for knock out. This could be "singular" "combinatorial"""

def generate_mutant_seq(wt_seq, mode, mut, alt_AA = ""):
    """This mode will generate the singular mutations. Each mutant produced by this has ONE site 
     replaced with the new amino acid, informed by the mutant manifest"""

    mut_list = []

    if (mode == "singular"):
        alternative_aminos = list(alt_AA)
        #change each residue and append 
        for r in mut:
            for A in alternative_aminos:
                change_residue_to_A(residue_number, OG_AA, New_AA, wt_seq)



        """Currently implemented to make homogenous changes (all mutation sites are same)"""
        if (mode == "combinatorial"):
                iteration_length = len(mut)
                alternative_aminos = list(alt_AA)
                #change each residue and append 

                for A in alternative_aminos:
                    i = 0
                    while (i < iteration_length):

return mut_list

def change_residue_to_A(residue,OG_AA, new_AA, wt_seq):
    """This will modify one residue in a wet_seq"""
    new_mut = list(wt_seq)
    #TODO do I need to up my assert game?
    assert(new_mut[residue]== OG_AA)
    new_mut[residue] = new_AA
    return str(new_mut)

def change_multiple_to_A_list(residue, OG_AA, new_AA, wt_seq):
    """This will modify multiple residues in a wet_seq. residue, OG_AA, 
    new_AA should all be equally lengthed lists"""
    assert(len(residue)== len(OG_AA))
    new_mut = list(wt_seq)
    for r in residue:
    #TODO do I need to up my assert game?
        assert(new_mut[r]== OG_AA)
        new_mut[residue] = new_AA
    return str(new_mut)


def process_mut(mut)-> pd.DataFrame:
    for x in mut: 
    new_mut = list(wt_seq)
            OG_AA = mut[r][0]
            New_AA = alternative_aminos[A]
            residue_number = mut[r][1:-2]