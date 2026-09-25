#TODO: look over
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def normalize_sequence(sequence: str) -> str:
    """
    Normalize and validate a protein sequence.
    """
    normalized = "".join(sequence.split()).upper()
    AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

    if not normalized:
        raise ValueError("Protein sequence is empty.")

    invalid = sorted(set(normalized) - AMINO_ACIDS)
    if invalid:
        raise ValueError(
            f"Protein sequence contains unsupported residues: {invalid}"
        )

    return normalized

def write_fasta(protein_id: str, sequence: str, destination: str | Path,) -> Path:
    """
    Write one protein sequence as a FASTA file.
    """
    if not protein_id.strip():
        raise ValueError("protein_id cannot be empty.")

    if "\n" in protein_id or "\r" in protein_id:
        raise ValueError("protein_id cannot contain line breaks.")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    wrapped_sequence = "\n".join(
        sequence[start:start + 80]
        for start in range(0, len(sequence), 80)
    )

    destination.write_text(
        f">{protein_id}\n{wrapped_sequence}\n",
        encoding="utf-8",
    )

    return destination


def prepare_wt_input(sequence: str, protein_id: str, input_directory: Path,) -> dict:
    """
    Convert user input into the WT FASTA and store it on the colabs local disk
    """
    
    normalized = normalize_sequence(sequence)
    wt_fasta = input_directory / "wt_input.fasta"

    write_fasta(
        protein_id=protein_id,
        sequence=normalized,
        destination=wt_fasta,
    )

    return {
        "protein_id": protein_id,
        "sequence": normalized,
        "sequence_length": len(normalized),
        "wt_fasta_path": wt_fasta,
    }
#TODO: assert QC

