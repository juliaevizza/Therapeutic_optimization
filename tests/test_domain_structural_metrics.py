from __future__ import annotations

from pathlib import Path

import numpy as np

from therapeutic_optimization.structural_analysis.metrics import analyze_structure_pair


def _write_ca_structure(path: Path, coordinates: list[tuple[float, float, float]]) -> None:
    lines = []
    for index, (x, y, z) in enumerate(coordinates, start=1):
        lines.append(
            f'ATOM  {index:5d}  CA  ALA A{index:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C'
        )
    path.write_text('\n'.join([*lines, 'TER', 'END', '']), encoding='utf-8')


def test_binder_rigid_body_motion_does_not_change_binder_or_core_rmsd(tmp_path):
    binder = [(0, 5, 0), (1, 6, 0), (2, 5, 1), (3, 6, 1)]
    core = [(0, 0, 0), (2, 0, 1), (4, 1, 0), (6, 0, 1), (8, 1, 0), (10, 0, 1)]
    moved_binder = [(x + 20, y + 10, z - 4) for x, y, z in binder]
    wt_path = tmp_path / 'wt.pdb'
    mutant_path = tmp_path / 'mutant.pdb'
    _write_ca_structure(wt_path, [*binder, *core])
    _write_ca_structure(mutant_path, [*moved_binder, *core])

    metrics = analyze_structure_pair(
        wt_path,
        mutant_path,
        variant_id='K7R',
        mutation_spec='K7R',
        per_residue_dir=tmp_path / 'per_residue',
        figure_dir=tmp_path / 'figures',
        binder_range=(1, 4),
    )

    assert metrics['global_ca_rmsd'] > 1.0
    assert np.isclose(metrics['core_ca_rmsd'], 0.0, atol=1e-5)
    assert np.isclose(metrics['binder_ca_rmsd'], 0.0, atol=1e-5)
    assert np.isclose(metrics['mutation_ca_displacement_max'], 0.0, atol=1e-5)
    assert np.isclose(metrics['intradomain_contact_change_fraction'], 0.0, atol=1e-5)
