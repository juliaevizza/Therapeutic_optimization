from __future__ import annotations

from itertools import combinations, product
from math import prod
from pathlib import Path

import pandas as pd

from ..config import MutationConfig, ProjectPaths


#TODO generate all experimentally interesting data structures of mutants 
"""This defintion encompasses all of the possible modes of generate new mutants based on the models 
predicted sites for knock out. This could be "singular" "combinatorial"""

def generate_mutant_seq(wt, mode, mut, alt_AA = ""):
    """This mode will generate the singular mutations. Each mutant produced by this has ONE site 
     replaced with the new amino acid, informed by the mutant manifest"""

    mut_list = []


    if (mode == "singular"):
        alternative_aminos = list(alt_AA)
        #change each residue and append 
        for r in mut:
            for A in alternative_aminos:
                new_mut = list(wt)
                OG_AA = mut[r][0]
                New_AA = alternative_aminos[A]
                residue_number = mut[r][1:-2]

            #TODO do I need to up my assert game?
            assert(new_mut[r]== OG_AA)
            new_mut[residue_number] = New_AA
            str(new_mut)
            mut_list.append(new_mut)



        """Currently implemented to make homogenous changes (all mutation sites are same)"""
        if (mode == "combinatorial"):
                iteration_length = len(mut)
                alternative_aminos = list(alt_AA)
                #change each residue and append 

                for A in alternative_aminos:
                    i = 0
                    while (i < iteration_length):


#TODO: pull out mutate one site code, then rethread singular mode to call it mulitple times
# then thread combinatorial mode to call it it and perform essentially the SDM protocol 


    return mut_list


