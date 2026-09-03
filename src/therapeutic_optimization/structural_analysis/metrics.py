from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser, Superimposer

from ..io import parse_mutation

RESIDUE_VOLUME = {
    'A': 67.0, 'R': 148.0, 'N': 96.0, 'D': 91.0, 'C': 86.0,
    'Q': 114.0, 'E': 109.0, 'G': 48.0, 'H': 118.0, 'I': 124.0,
    'L': 124.0, 'K': 135.0, 'M': 124.0, 'F': 135.0, 'P': 90.0,
    'S': 73.0, 'T': 93.0, 'W': 163.0, 'Y': 141.0, 'V': 105.0,
}

DISPLACEMENT_Y_LIMITS = (0.0, 50.0)


def load_structure(path: str | Path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.pdb':
        parser = PDBParser(QUIET=True)
    elif suffix in {'.cif', '.mmcif'}:
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f'Unsupported structure format: {path}')
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    return model


def ca_atoms_by_position(model) -> dict[int, object]:
    atoms: dict[int, object] = {}
    for chain in model:
        for residue in chain:
            hetflag, resseq, _icode = residue.id
            if hetflag.strip():
                continue
            if 'CA' in residue:
                if int(resseq) in atoms:
                    raise ValueError(
                        'Multiple chains share residue numbers; this single-chain structural metric '
                        'implementation requires unique residue numbering.'
                    )
                atoms[int(resseq)] = residue['CA']
    if not atoms:
        raise ValueError('No C-alpha atoms were found in the structure.')
    return atoms


def matched_positions(wt_model, mutant_model) -> list[int]:
    wt = ca_atoms_by_position(wt_model)
    mut = ca_atoms_by_position(mutant_model)
    positions = sorted(set(wt) & set(mut))
    if len(positions) < 3:
        raise ValueError('Fewer than three matched C-alpha positions were found.')
    return positions


def _positions_outside_range(positions: Iterable[int], excluded_range: tuple[int, int] | None) -> list[int]:
    if excluded_range is None:
        return list(positions)
    start, end = excluded_range
    return [position for position in positions if not start <= position <= end]


def ca_rmsd(wt_model, mutant_model, positions: Iterable[int] | None = None) -> float:
    """Return C-alpha RMSD after a least-squares fit over the selected residues."""
    wt = ca_atoms_by_position(wt_model)
    mut = ca_atoms_by_position(mutant_model)
    available = matched_positions(wt_model, mutant_model)
    positions = available if positions is None else sorted(set(positions) & set(available))
    if len(positions) < 3:
        raise ValueError('Fewer than three matched C-alpha positions were selected for alignment.')
    fixed = [wt[p] for p in positions]
    moving = [mut[p] for p in positions]
    superimposer = Superimposer()
    superimposer.set_atoms(fixed, moving)
    return float(superimposer.rms)


def align_mutant_to_wt(
    wt_model,
    mutant_model,
    positions: Iterable[int] | None = None,
) -> float:
    wt = ca_atoms_by_position(wt_model)
    mut = ca_atoms_by_position(mutant_model)
    available = matched_positions(wt_model, mutant_model)
    positions = available if positions is None else sorted(set(positions) & set(available))
    if len(positions) < 3:
        raise ValueError('Fewer than three matched C-alpha positions were selected for alignment.')
    fixed = [wt[p] for p in positions]
    moving = [mut[p] for p in positions]
    superimposer = Superimposer()
    superimposer.set_atoms(fixed, moving)
    superimposer.apply(list(mutant_model.get_atoms()))
    return float(superimposer.rms)


def residue_displacements(wt_model, mutant_model) -> pd.DataFrame:
    wt = ca_atoms_by_position(wt_model)
    mut = ca_atoms_by_position(mutant_model)
    positions = matched_positions(wt_model, mutant_model)
    return pd.DataFrame(
        {
            'position': positions,
            'ca_displacement': [float(np.linalg.norm(wt[p].coord - mut[p].coord)) for p in positions],
        }
    )


def radius_of_gyration(model) -> float:
    coords = np.asarray([atom.coord for atom in model.get_atoms() if atom.element != 'H'], dtype=float)
    if len(coords) == 0:
        raise ValueError('No heavy atoms available for radius of gyration.')
    centroid = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))


def mean_ca_plddt(model) -> float:
    values = [float(atom.bfactor) for atom in ca_atoms_by_position(model).values()]
    return float(np.mean(values))


def contact_pairs(model, cutoff: float = 8.0) -> set[tuple[int, int]]:
    atoms = ca_atoms_by_position(model)
    positions = sorted(atoms)
    coords = np.asarray([atoms[p].coord for p in positions], dtype=float)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    pairs: set[tuple[int, int]] = set()
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if distances[i, j] <= cutoff:
                pairs.add((positions[i], positions[j]))
    return pairs


def local_contact_pairs(model, position: int, cutoff: float = 8.0) -> set[int]:
    atoms = ca_atoms_by_position(model)
    if position not in atoms:
        raise ValueError(f'Mutation position {position} is absent from structure.')
    center = atoms[position].coord
    return {
        other_position
        for other_position, atom in atoms.items()
        if other_position != position and float(np.linalg.norm(center - atom.coord)) <= cutoff
    }


def mutation_distance_from_centroid(model, position: int) -> float:
    atoms = ca_atoms_by_position(model)
    if position not in atoms:
        raise ValueError(f'Mutation position {position} is absent from structure.')
    coords = np.asarray([atom.coord for atom in atoms.values()], dtype=float)
    centroid = coords.mean(axis=0)
    return float(np.linalg.norm(atoms[position].coord - centroid))


def _plot_displacement(df: pd.DataFrame, variant_id: str, positions: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylim(0, 50)
    ax.plot(df['position'], df['ca_displacement'])
    for position in positions:
        ax.axvline(position, linestyle='--')
    ax.set_xlabel('Residue position')
    ax.set_ylabel('Cα displacement after alignment (Å)')
    ax.set_ylim(*DISPLACEMENT_Y_LIMITS)
    ax.set_title(f'WT vs {variant_id}')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def analyze_structure_pair(
    wt_structure_path: str | Path,
    mutant_structure_path: str | Path,
    variant_id: str,
    mutation_spec: str,
    per_residue_dir: str | Path,
    figure_dir: str | Path,
    local_window: int = 5,
    contact_cutoff: float = 8.0,
    binder_range: tuple[int, int] | None = None,
) -> dict:
    """Compare one mutant model to WT and return interpretable preservation metrics."""
    wt_model = load_structure(wt_structure_path)
    mutant_model = load_structure(mutant_structure_path)

    mutations = [item for item in mutation_spec.split(';') if item]
    parsed = [parse_mutation(item) for item in mutations]
    mutation_positions = [position for _wt, position, _mut in parsed]

    positions = matched_positions(wt_model, mutant_model)
    global_ca_rmsd = ca_rmsd(wt_model, mutant_model)
    binder_positions = (
        [p for p in positions if binder_range[0] <= p <= binder_range[1]]
        if binder_range is not None else []
    )
    core_positions = _positions_outside_range(positions, binder_range)
    binder_ca_rmsd = ca_rmsd(wt_model, mutant_model, binder_positions) if binder_range else np.nan
    core_ca_rmsd = align_mutant_to_wt(wt_model, mutant_model, core_positions)
    displacement = residue_displacements(wt_model, mutant_model)

    site_displacements = displacement.loc[
        displacement['position'].isin(mutation_positions), 'ca_displacement'
    ]
    if len(site_displacements) != len(mutation_positions):
        missing = sorted(set(mutation_positions) - set(displacement['position'].astype(int)))
        raise ValueError(f'Mutation positions absent from aligned structures: {missing}')

    local_mask = np.zeros(len(displacement), dtype=bool)
    for position in mutation_positions:
        local_mask |= displacement['position'].between(position - local_window, position + local_window).to_numpy()
    local = displacement.loc[local_mask]

    wt_rg = radius_of_gyration(wt_model)
    mutant_rg = radius_of_gyration(mutant_model)
    wt_contacts = contact_pairs(wt_model, contact_cutoff)
    mutant_contacts = contact_pairs(mutant_model, contact_cutoff)
    lost = wt_contacts - mutant_contacts
    gained = mutant_contacts - wt_contacts
    retained = wt_contacts & mutant_contacts
    union = wt_contacts | mutant_contacts
    contact_change_fraction = (len(lost) + len(gained)) / max(1, len(union))

    def is_intradomain(pair: tuple[int, int]) -> bool:
        if binder_range is None:
            return True
        start, end = binder_range
        first_is_binder = start <= pair[0] <= end
        second_is_binder = start <= pair[1] <= end
        return first_is_binder == second_is_binder

    wt_intradomain = {pair for pair in wt_contacts if is_intradomain(pair)}
    mutant_intradomain = {pair for pair in mutant_contacts if is_intradomain(pair)}
    intradomain_union = wt_intradomain | mutant_intradomain
    intradomain_changes = wt_intradomain ^ mutant_intradomain
    intradomain_contact_change_fraction = len(intradomain_changes) / max(1, len(intradomain_union))

    local_lost_total = 0
    local_gained_total = 0
    for position in mutation_positions:
        wt_local = local_contact_pairs(wt_model, position, contact_cutoff)
        mut_local = local_contact_pairs(mutant_model, position, contact_cutoff)
        local_lost_total += len(wt_local - mut_local)
        local_gained_total += len(mut_local - wt_local)

    wt_centroid_distances = [mutation_distance_from_centroid(wt_model, p) for p in mutation_positions]
    mut_centroid_distances = [mutation_distance_from_centroid(mutant_model, p) for p in mutation_positions]
    volume_changes = [RESIDUE_VOLUME[mut] - RESIDUE_VOLUME[wt] for wt, _p, mut in parsed]

    per_residue_dir = Path(per_residue_dir)
    figure_dir = Path(figure_dir)
    per_residue_path = per_residue_dir / f'{variant_id}_per_residue_displacement.csv'
    plot_path = figure_dir / f'{variant_id}_displacement.png'
    per_residue_path.parent.mkdir(parents=True, exist_ok=True)
    displacement.to_csv(per_residue_path, index=False)
    _plot_displacement(displacement, variant_id, mutation_positions, plot_path)

    wt_plddt = mean_ca_plddt(wt_model)
    mutant_plddt = mean_ca_plddt(mutant_model)

    return {
        'variant_id': variant_id,
        'mutation_spec': mutation_spec,
        'mutation_count': len(mutations),
        'global_ca_rmsd': global_ca_rmsd,
        'core_ca_rmsd': core_ca_rmsd,
        'binder_ca_rmsd': float(binder_ca_rmsd),
        'mean_ca_displacement': float(displacement['ca_displacement'].mean()),
        'median_ca_displacement': float(displacement['ca_displacement'].median()),
        'max_ca_displacement': float(displacement['ca_displacement'].max()),
        'mutation_ca_displacement': float(site_displacements.mean()),
        'mutation_ca_displacement_max': float(site_displacements.max()),
        'local_window': local_window,
        'local_mean_ca_displacement': float(local['ca_displacement'].mean()),
        'local_median_ca_displacement': float(local['ca_displacement'].median()),
        'local_max_ca_displacement': float(local['ca_displacement'].max()),
        'wt_radius_of_gyration': wt_rg,
        'mutant_radius_of_gyration': mutant_rg,
        'radius_of_gyration_change': mutant_rg - wt_rg,
        'wt_mean_plddt': wt_plddt,
        'mutant_mean_plddt': mutant_plddt,
        'mean_plddt_change': mutant_plddt - wt_plddt,
        'global_contacts_lost': len(lost),
        'global_contacts_gained': len(gained),
        'global_contacts_retained': len(retained),
        'global_contact_changes': len(lost) + len(gained),
        'global_contact_change_fraction': float(contact_change_fraction),
        'intradomain_contact_change_fraction': float(intradomain_contact_change_fraction),
        'local_contacts_lost': int(local_lost_total),
        'local_contacts_gained': int(local_gained_total),
        'mean_wt_centroid_distance': float(np.mean(wt_centroid_distances)),
        'mean_mutant_centroid_distance': float(np.mean(mut_centroid_distances)),
        'mean_centroid_distance_change': float(np.mean(np.asarray(mut_centroid_distances) - np.asarray(wt_centroid_distances))),
        'mean_residue_volume_change': float(np.mean(volume_changes)),
        'per_residue_csv': str(per_residue_path),
        'displacement_plot': str(plot_path),
        'wt_structure': str(Path(wt_structure_path)),
        'mutant_structure': str(Path(mutant_structure_path)),
    }
