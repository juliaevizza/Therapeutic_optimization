from __future__ import annotations

import pandas as pd


#TODO generate all experimentally interesting data structures of mutants 
"""This defintion encompasses all of the possible modes of generate new mutants based on the models 
predicted sites for knock out. This could be "singular" """


def generate_mutant_seq(wt_seq, mode, mut, alt_AA = ["R"]):
    """This mode will generate the singular mutations. Each mutant produced by this has ONE site 
     replaced with the new amino acid, informed by the mut argument"""

### ASSSERTTSSSSSS
    assert(type(wt_seq) == str)
    assert(type(mode) == str)
    assert isinstance(mut, list)
    assert all(isinstance(item, str) for item in mut)
    assert(type(alt_AA) == str)


    parsed_mut = parse_mut(mut)
    OG_AA = parsed_mut["OG_AA"]
    residue_number = parsed_mut["Residue_number"]
    alternative_aminos = list(alt_AA)
    new_seq = []

    if (mode == "singular"):
        #changes singular mutation sites to each of the aminos of interest
        for r in range(len(mut)):
            for A in alternative_aminos:
                new_seq.append(change_residue_to_A(residue_number[r], OG_AA[r], A, wt_seq))

    if (mode == "all_homogenous"):
        #changes every mutation site (for each amino of interest)
        for A in alternative_aminos:
                new_seq.append(change_multiple_to_A(residue_number, OG_AA, A, wt_seq))
         
    if (mode == "combinatorial"):
        #generate mutant array
        #while i < len(alternative_aminos)^2:
        #new_seq.append(wt_seq)

        branches = [list(wt_seq)]
        n_sites = len(mut)
        n = 0
        while n < n_sites:
            b_length = len(branches)
            b = 0
            new_branches = []
            while b < b_length:
                for A in alternative_aminos:
                    seq = list(branches[b])
                    seq[residue_number[n]-1] = A
                    if (n == (n_sites)- 1):
                        new_seq.append("".join(seq))
                    new_branches.append(seq)
                b = b + 1
            branches = new_branches
            n = n + 1
    return new_seq

def change_residue_to_A(r, OG_AA, new_AA, wt_seq):
    """This will modify one residue in a wet_seq"""
    new_mut = list(wt_seq)
    #TODO do I need to up my assert game?
    assert(new_mut[r-1] == OG_AA)
    new_mut[r - 1] = new_AA
    return "".join(new_mut)

def change_multiple_to_A(residues, OG_AA, new_AA, wt_seq):
    """This will modify multiple residues in a wet_seq. residue, OG_AA, 
    new_AA should all be equally lengthed lists"""
    assert(len(residues)== len(OG_AA))

    new_mut = list(wt_seq)
    for r, og_aa in zip(residues, OG_AA):
    #TODO do I need to up my assert game?
        assert(new_mut[r-1]== og_aa)
        new_mut[r-1] = new_AA
    return "".join(new_mut)

def parse_mut(mut)-> pd.DataFrame:
    """This will parse the string list output of the model to be well formatted 
    for mutant generation."""
    assert isinstance(mut, list)
    assert all(isinstance(item, str) for item in mut)    
    OG_AA = []
    residue_number = []
    for i in mut: 
            OG_AA.append(i[0])
            residue_number.append(int(i[1:]))
    data = {'OG_AA' : OG_AA, 'Residue_number' : residue_number,}
    mutant_list = pd.DataFrame(data)
    return mutant_list

#TODO: assert QC
def assertQC():
    pass