#TODO: look over
from __future__ import annotations

from datetime import datetime, timezone

from ..config import ProjectPaths
from ..io import normalize_sequence, write_fasta, write_json


def prepare_wt_input(
    sequence: str,
    protein_id: str,
    paths: ProjectPaths,
) -> dict:
    """T1: convert user sequence input into the canonical WT FASTA."""
    paths.ensure()
    normalized = normalize_sequence(sequence)
    write_fasta(protein_id, normalized, paths.wt_fasta)
    metadata = {
        'protein_id': protein_id,
        'sequence_length': len(normalized),
        'wt_fasta': str(paths.wt_fasta),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(metadata, paths.input_metadata)
    return metadata
