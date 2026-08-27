from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
MUTATION_PATTERN = re.compile(r'^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$')


def normalize_sequence(sequence: str) -> str:
    sequence = ''.join(sequence.split()).upper()
    if not sequence:
        raise ValueError('Protein sequence is empty.')
    invalid = sorted(set(sequence) - AMINO_ACIDS)
    if invalid:
        raise ValueError(f'Protein sequence contains unsupported residues: {invalid}')
    return sequence


def write_fasta(protein_id: str, sequence: str, path: str | Path) -> Path:
    sequence = normalize_sequence(sequence)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        handle.write(f'>{protein_id}\n')
        for start in range(0, len(sequence), 80):
            handle.write(sequence[start:start + 80] + '\n')
    return path


def read_single_fasta(path: str | Path) -> tuple[str, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    header: str | None = None
    chunks: list[str] = []
    records = 0
    with path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('>'):
                records += 1
                if records > 1:
                    raise ValueError(f'Expected exactly one FASTA record in {path}.')
                header = line[1:].strip() or path.stem
            else:
                if records == 0:
                    raise ValueError(f'FASTA sequence appeared before a header in {path}.')
                chunks.append(line)
    if records != 1:
        raise ValueError(f'Expected exactly one FASTA record in {path}.')
    return header or path.stem, normalize_sequence(''.join(chunks))


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    match = MUTATION_PATTERN.fullmatch(mutation.strip().upper())
    if not match:
        raise ValueError(f'Invalid mutation notation: {mutation!r}. Expected e.g. K34A.')
    wt_aa, position_text, mutant_aa = match.groups()
    position = int(position_text)
    if position < 1:
        raise ValueError('Mutation positions are 1-based and must be >= 1.')
    return wt_aa, position, mutant_aa


def apply_mutations(sequence: str, mutations: Iterable[str]) -> str:
    sequence = normalize_sequence(sequence)
    output = list(sequence)
    seen_positions: set[int] = set()
    for mutation in mutations:
        wt_aa, position, mutant_aa = parse_mutation(mutation)
        if position > len(sequence):
            raise ValueError(f'{mutation}: position {position} exceeds sequence length {len(sequence)}.')
        if position in seen_positions:
            raise ValueError(f'Multiple mutations target position {position}.')
        if sequence[position - 1] != wt_aa:
            raise ValueError(
                f'{mutation}: expected WT residue {wt_aa} at position {position}, '
                f'but sequence contains {sequence[position - 1]}.'
            )
        output[position - 1] = mutant_aa
        seen_positions.add(position)
    return ''.join(output)


def write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    return path
