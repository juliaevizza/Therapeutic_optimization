from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ProjectPaths, WorkflowConfig
from .ranking import run_r1, run_r2
from .structural_analysis import build_structure_predictor, run_s1
from .transformations import generate_mutant_manifest, prepare_wt_input
from .ubiquitination_prediction import build_predictor, run_ub2, run_up1


class OptimizationWorkflow:
    """Top-level orchestration with explicit T1/UP1/T2/S1/R1/UB2/R2 stages."""

    def __init__(
        self,
        project_root: str | Path,
        config: WorkflowConfig | None = None,
    ) -> None:
        self.paths = ProjectPaths.from_root(project_root)
        self.paths.ensure()
        self.config = config or WorkflowConfig()
        self._ubi_predictor = None
        self._structure_predictor = None

    @property
    def ubi_predictor(self):
        if self._ubi_predictor is None:
            self._ubi_predictor = build_predictor(self.config.ubiquitination)
        return self._ubi_predictor

    @property
    def structure_predictor(self):
        if self._structure_predictor is None:
            self._structure_predictor = build_structure_predictor(self.config.structure)
        return self._structure_predictor

    def T1(self, sequence: str, protein_id: str = 'WT') -> dict:
        return prepare_wt_input(sequence, protein_id, self.paths)

    def UP1(self) -> pd.DataFrame:
        return run_up1(self.ubi_predictor, self.paths)

    def T2(self, up1: pd.DataFrame | None = None) -> pd.DataFrame:
        if up1 is None:
            up1 = pd.read_csv(self.paths.tables / 'UP1_wt_ubiquitination.csv')
        return generate_mutant_manifest(up1, self.paths, self.config.mutation)

    def S1(
        self,
        manifest: pd.DataFrame | None = None,
        predict_structures: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if manifest is None:
            manifest = pd.read_csv(self.paths.tables / 'T2_mutation_manifest.csv')
        return run_s1(
            manifest=manifest,
            paths=self.paths,
            predictor=self.structure_predictor,
            thresholds=self.config.structural_thresholds,
            predict_structures=predict_structures,
        )

    def R1(
        self,
        manifest: pd.DataFrame | None = None,
        s1_metrics: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if manifest is None:
            manifest = pd.read_csv(self.paths.tables / 'T2_mutation_manifest.csv')
        if s1_metrics is None:
            s1_metrics = pd.read_csv(self.paths.tables / 'S1_structural_metrics.csv')
        return run_r1(manifest, s1_metrics, self.paths)

    def UB2(self, s1_conserved: pd.DataFrame | None = None) -> pd.DataFrame:
        if s1_conserved is None:
            s1_conserved = pd.read_csv(self.paths.tables / 'S1_structurally_conserved.csv')
        return run_ub2(self.ubi_predictor, s1_conserved, self.paths)

    def R2(
        self,
        up1: pd.DataFrame | None = None,
        ub2: pd.DataFrame | None = None,
        s1_conserved: pd.DataFrame | None = None,
    ):
        if up1 is None:
            up1 = pd.read_csv(self.paths.tables / 'UP1_wt_ubiquitination.csv')
        if ub2 is None:
            ub2 = pd.read_csv(self.paths.tables / 'UB2_mutant_ubiquitination.csv')
        if s1_conserved is None:
            s1_conserved = pd.read_csv(self.paths.tables / 'S1_structurally_conserved.csv')
        return run_r2(up1, ub2, s1_conserved, self.paths)

    def run_all(
        self,
        sequence: str,
        protein_id: str = 'WT',
        predict_structures: bool = True,
    ) -> dict[str, object]:
        """Execute the complete workflow. GPU-heavy stages still fail loudly if dependencies are missing."""
        t1 = self.T1(sequence, protein_id)
        up1 = self.UP1()
        t2 = self.T2(up1)
        s1_metrics, s1_conserved = self.S1(t2, predict_structures=predict_structures)
        r1 = self.R1(t2, s1_metrics)
        ub2 = self.UB2(s1_conserved)
        r2_all, r2_optimized, r2_needs = self.R2(up1, ub2, s1_conserved)
        return {
            'T1': t1,
            'UP1': up1,
            'T2': t2,
            'S1_metrics': s1_metrics,
            'S1_conserved': s1_conserved,
            'R1': r1,
            'UB2': ub2,
            'R2_all': r2_all,
            'R2_optimized': r2_optimized,
            'R2_needs_further_optimization': r2_needs,
        }
