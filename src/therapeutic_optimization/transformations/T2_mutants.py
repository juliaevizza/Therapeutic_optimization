from __future__ import annotations
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import pandas as pd


#TODO generate all experimentally interesting data structures of mutants 
"""This defintion encompasses all of the possible modes of generate new mutants based on the models 
predicted sites for knock out. This could be "singular" """


def generate_mutant_seq(wt_seq, mode, mut, mutant_manifest, alt_AA = ["R"]):
    """This mode will generate the singular mutations. Each mutant produced by this has ONE site 
     replaced with the new amino acid, informed by the mut argument. It will return a string"""

### ASSSERTTSSSSSS
    assert(type(wt_seq) == str)
    assert(type(mode) == str)
    assert isinstance(mut, list)
    assert all(isinstance(item, str) for item in mut)
    for i in alt_AA:
        assert(type(i) == str)

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

    ##write new info (sequences and ids) to the mutant_manifest
    for sequence in new_seq:
        mutant_id = make_mutant_id(wt_seq, sequence)
        mutant_manifest.loc[len(mutant_manifest)] = [mutant_id, sequence, pd.NA, pd.NA, pd.NA]

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

def make_mutant_id(wt_seq, mutant_seq):
    """This function will take the mutant sequence and generate the shorthand mutant 
    id for it by comaparing which positions are different and than formatting it"""
    return "_".join(
        f"{old}{position}{new}" for position, (old, new) in enumerate(zip(wt_seq, mutant_seq), start=1) if old != new
    )

def write_mutant_fastas(manifest, output_directory):
    """This function will write the mutant fastas into the drive for the colabfold
     command line interface."""
    for row in manifest.itertuples():
        if row.mutant_id == "WT":
            continue
        SeqIO.write(SeqRecord(Seq(row.sequence), id=row.mutant_id, description=""), output_directory / f"{row.mutant_id}.fasta", "fasta")
        manifest.at[row.Index, "path"] = str(output_directory / f"{row.mutant_id}.fasta")