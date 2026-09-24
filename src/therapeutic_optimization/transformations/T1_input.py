#TODO: look over
from __future__ import annotations

from datetime import datetime, timezone

def prepare_wt_input(
    sequence: str,
    protein_id: str,
) -> dict:
    """T1: convert user sequence input into the canonical WT FASTA."""
    normalized = normalize_sequence(sequence)
    write_fasta(protein_id, normalized, 
                ##TODO call where we want the wt fasta to sit
                .wt_fasta)
    ## save fasta to storage/fastas
    metadata = {
        'protein_id': protein_id,
        'sequence_length': len(normalized),
        'wt_fasta': str(paths.wt_fasta),
    }
    write_json(metadata, paths.input_metadata)
    return metadata

#TODO: assert QC

