from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import pandas as pd

from therapeutic_optimization.structural_analysis.metrics import _plot_displacement


def test_displacement_plot_uses_fixed_zero_to_fifty_angstrom_y_axis(tmp_path, monkeypatch):
    saved_y_limits = []

    def capture_y_limits(figure, _path, **_kwargs):
        saved_y_limits.append(figure.axes[0].get_ylim())

    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', capture_y_limits)
    displacement = pd.DataFrame(
        {
            'position': [1, 2, 3],
            'ca_displacement': [0.5, 12.0, 49.5],
        }
    )

    _plot_displacement(displacement, 'K2R', [2], Path(tmp_path / 'plot.png'))

    assert saved_y_limits == [(0.0, 50.0)]
