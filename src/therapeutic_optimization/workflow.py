from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ProjectPaths, WorkflowConfig
from .comparison import build_lysine_free_comparison
from .complex_search import run_complex_search, esm2_gate
from .esm2_analysis import ESM2Scorer, run_esm2_analysis
from .ranking import run_r1, run_r2
from .io import read_single_fasta
from .structural_analysis.colabfold import find_rank1_structure
from .structural_analysis import build_structure_predictor, run_s1
from .transformations import (
    generate_all_lysine_to_arginine_manifest,
    generate_mutant_manifest,
    prepare_wt_input,
)
from .ubiquitination_prediction import build_predictor, run_ub2, run_up1


class OptimizationWorkflow:
    """Top-level orchestration with explicit T1/UP1/T2/ESM2/S1/R1/UB2/R2 stages."""

    def __init__(
        self,
        project_root: str | Path,
        config: WorkflowConfig | None = None,
    ) -> None:
        self.config = config or WorkflowConfig()
        if self.config.mode not in {'basic', 'sophisticated'}:
            raise ValueError("Workflow mode must be 'basic' or 'sophisticated'.")
        self.stage_suffix = 'basic' if self.config.mode == 'basic' else 'complex'
        self.paths = ProjectPaths.from_root(project_root, self.stage_suffix)
        self.paths.ensure()
        self._complex_results = None
        self._ubi_predictor = None
        self._structure_predictor = None
        self._esm2_scorer = None

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

    @property
    def esm2_scorer(self) -> ESM2Scorer:
        if self._esm2_scorer is None:
            self._esm2_scorer = ESM2Scorer(self.config.esm2)
        return self._esm2_scorer

    def T1_basic(self, sequence: str, protein_id: str = 'WT') -> dict:
        self._complex_results = None
        return prepare_wt_input(sequence, protein_id, self.paths)

    def UP1_basic(self) -> pd.DataFrame:
        return run_up1(self.ubi_predictor, self.paths)

    def T2_basic(self, up1: pd.DataFrame | None = None) -> pd.DataFrame:
        if up1 is None:
            up1 = pd.read_csv(self.paths.table('UP1_wt_ubiquitination.csv'))
        return generate_mutant_manifest(up1, self.paths, self.config.mutation)

    def T2_all_lysine_to_arginine(self) -> pd.DataFrame:
        return generate_all_lysine_to_arginine_manifest(self.paths)

    def ESM2_basic(self, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
        """Compare T2 mutants with WT using pseudo-perplexity and embeddings."""
        if manifest is None:
            manifest = pd.read_csv(self.paths.table('T2_mutation_manifest.csv'))
        return run_esm2_analysis(
            manifest,
            self.paths,
            scorer=self.esm2_scorer,
        )

    def S1_basic(
        self,
        manifest: pd.DataFrame | None = None,
        predict_structures: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if manifest is None:
            manifest = pd.read_csv(self.paths.table('T2_mutation_manifest.csv'))
        return run_s1(
            manifest=manifest,
            paths=self.paths,
            predictor=self.structure_predictor,
            thresholds=self.config.structural_thresholds,
            predict_structures=predict_structures,
        )

    def R1_basic(
        self,
        manifest: pd.DataFrame | None = None,
        s1_metrics: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if manifest is None:
            manifest = pd.read_csv(self.paths.table('T2_mutation_manifest.csv'))
        if s1_metrics is None:
            s1_metrics = pd.read_csv(self.paths.table('S1_structural_metrics.csv'))
        return run_r1(manifest, s1_metrics, self.paths)

    def UB2_basic(self, s1_conserved: pd.DataFrame | None = None) -> pd.DataFrame:
        if s1_conserved is None:
            s1_conserved = pd.read_csv(self.paths.table('S1_structurally_conserved.csv'))
        return run_ub2(self.ubi_predictor, s1_conserved, self.paths)

    def R2_basic(
        self,
        up1: pd.DataFrame | None = None,
        ub2: pd.DataFrame | None = None,
        s1_conserved: pd.DataFrame | None = None,
        esm2_results: pd.DataFrame | None = None,
        use_saved_esm2: bool = True,
    ):
        if up1 is None:
            up1 = pd.read_csv(self.paths.table('UP1_wt_ubiquitination.csv'))
        if ub2 is None:
            ub2 = pd.read_csv(self.paths.table('UB2_mutant_ubiquitination.csv'))
        if s1_conserved is None:
            s1_conserved = pd.read_csv(self.paths.table('S1_structurally_conserved.csv'))
        if esm2_results is None and use_saved_esm2:
            esm2_path = self.paths.table('ESM2_mutant_comparison.csv')
            if esm2_path.exists():
                esm2_results = pd.read_csv(esm2_path)
        return run_r2(
            up1,
            ub2,
            s1_conserved,
            self.paths,
            esm2_results=esm2_results,
            esm2_config=self.config.esm2,
        )

    # Shared stages keep explicit names in both workflows.
    T1_complex = T1_basic
    UP1_complex = UP1_basic
    R1_complex = R1_basic
    UB2_complex = UB2_basic
    R2_complex = R2_basic

    def _dispatch(self, stage: str, *args, **kwargs):
        return getattr(self, f'{stage}_{self.stage_suffix}')(*args, **kwargs)

    def T1(self, *args, **kwargs):
        return self._dispatch('T1', *args, **kwargs)

    def UP1(self, *args, **kwargs):
        return self._dispatch('UP1', *args, **kwargs)

    def T2(self, up1=None, predict_structures: bool = True):
        if self.stage_suffix == 'complex':
            return self.T2_complex(up1, predict_structures=predict_structures)
        return self.T2_basic(up1)

    def ESM2(self, *args, **kwargs):
        return self._dispatch('ESM2', *args, **kwargs)

    def S1(self, *args, **kwargs):
        return self._dispatch('S1', *args, **kwargs)

    def R1(self, *args, **kwargs):
        return self._dispatch('R1', *args, **kwargs)

    def UB2(self, *args, **kwargs):
        return self._dispatch('UB2', *args, **kwargs)

    def R2(self, *args, **kwargs):
        return self._dispatch('R2', *args, **kwargs)

    def _cached_complex(self, manifest: pd.DataFrame | None) -> bool:
        if self._complex_results is None:
            return False
        return manifest is None or manifest.equals(self._complex_results['T2'])

    def T2_complex(
        self, up1: pd.DataFrame | None = None, predict_structures: bool = True,
    ) -> pd.DataFrame:
        """Interleave T2_complex → ESM2_complex → S1_complex until 25 pass."""
        from uuid import uuid4

        if up1 is None:
            up1 = pd.read_csv(self.paths.table('UP1_wt_ubiquitination.csv'))
        self._complex_results = None
        self.config.complex_search.validate()
        if self._ubi_predictor is not None and hasattr(self._ubi_predictor, 'release'):
            self._ubi_predictor.release()
        wt_score = None
        wt_structure = None
        batch_root = self.paths.storage / 'structures' / 'complex_batches' / uuid4().hex

        def score(candidate):
            nonlocal wt_score
            if wt_score is None:
                _, sequence = read_single_fasta(self.paths.wt_fasta)
                wt_score = self.esm2_scorer.score_sequence(sequence)
            return run_esm2_analysis(
                candidate, self.paths, scorer=self.esm2_scorer, wt_score=wt_score,
                output_path=self.paths.esm2 / 'complex_current_candidate.csv',
            )

        def structure(candidate):
            nonlocal wt_structure
            # ESM-2 can be reloaded for the next candidate; cache WT scores on CPU.
            self.esm2_scorer.release()
            result = run_s1(
                candidate, self.paths, self.structure_predictor,
                self.config.structural_thresholds, predict_structures=predict_structures,
                wt_structure_path=wt_structure,
                batch_output_dir=batch_root / str(candidate.iloc[0]['variant_id']),
                write_tables=False,
            )
            if wt_structure is None:
                wt_structure = find_rank1_structure(self.paths.structures_wt)
            return result

        try:
            self._complex_results = run_complex_search(
                up1, self.paths, self.config.complex_search,
                self.config.mutation.threshold, score, structure,
            )
        finally:
            if self._esm2_scorer is not None:
                self._esm2_scorer.release()
        return self._complex_results['T2'].copy()

    def ESM2_complex(self, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._cached_complex(manifest):
            return self._complex_results['ESM2'].copy()
        result = self.ESM2_basic(manifest)
        decisions = [esm2_gate(row, self.config.complex_search) for _, row in result.iterrows()]
        result['esm2_screen_pass'] = [decision[0] for decision in decisions]
        result['esm2_failure_reason'] = [decision[1] for decision in decisions]
        result.to_csv(self.paths.table('ESM2_mutant_comparison.csv'), index=False)
        return result

    def S1_complex(
        self, manifest: pd.DataFrame | None = None, predict_structures: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Reuse the adaptive search, or gate a supplied manifest before S1."""
        if self._cached_complex(manifest):
            return (self._complex_results['S1_metrics'].copy(),
                    self._complex_results['S1_conserved'].copy())
        if manifest is None:
            manifest = pd.read_csv(self.paths.table('T2_mutation_manifest.csv'))
        try:
            esm2 = self.ESM2_complex(manifest)
        finally:
            self.esm2_scorer.release()
        passing = set(esm2.loc[esm2['esm2_screen_pass'], 'variant_id'])
        eligible = manifest.loc[manifest['variant_id'].isin(passing)]
        metrics, conserved = self.S1_basic(eligible, predict_structures=predict_structures)
        rejected = manifest.loc[~manifest['variant_id'].isin(passing)].copy()
        rejected['analysis_status'] = 'SKIPPED_ESM2'
        rejected['analysis_error'] = None
        rejected['structure_pass'] = False
        rejected['structural_preservation_score'] = float('nan')
        reasons = esm2.set_index('variant_id')['esm2_failure_reason']
        rejected['structural_failure_reasons'] = rejected['variant_id'].map(reasons)
        metrics = pd.concat([metrics, rejected], ignore_index=True)
        metrics.to_csv(self.paths.table('S1_structural_metrics.csv'), index=False)
        return metrics, conserved

    def run_all(
        self,
        sequence: str,
        protein_id: str = 'WT',
        predict_structures: bool = True,
        analyze_esm2: bool = True,
    ) -> dict[str, object]:
        """Execute the complete workflow. GPU-heavy stages still fail loudly if dependencies are missing."""
        if self.config.mode == 'sophisticated' and not analyze_esm2:
            raise ValueError('Sophisticated mode requires ESM-2 screening.')
        t1 = self.T1(sequence, protein_id)
        up1 = self.UP1()
        t2 = self.T2(up1, predict_structures=predict_structures)
        esm2 = None
        if analyze_esm2:
            try:
                esm2 = self.ESM2(t2)
            finally:
                # This model is not needed again in run_all; release its memory
                # before structure prediction and later EUP mutant inference.
                self.esm2_scorer.release()
        s1_metrics, s1_conserved = self.S1(t2, predict_structures=predict_structures)
        r1 = self.R1(t2, s1_metrics)
        ub2 = self.UB2(s1_conserved)
        r2_all, r2_optimized, r2_needs = self.R2(
            up1,
            ub2,
            s1_conserved,
            esm2_results=esm2,
            use_saved_esm2=analyze_esm2,
        )
        results = {
            'T1': t1,
            'UP1': up1,
            'T2': t2,
            'ESM2': esm2,
            'S1_metrics': s1_metrics,
            'S1_conserved': s1_conserved,
            'R1': r1,
            'UB2': ub2,
            'R2_all': r2_all,
            'R2_optimized': r2_optimized,
            'R2_needs_further_optimization': r2_needs,
        }
        if self._complex_results is not None:
            results['search_summary'] = self._complex_results['search_summary']
        return results

    def run_lysine_free_comparison(
        self,
        sequence: str | None = None,
        protein_id: str = 'WT',
        predict_structures: bool = True,
    ) -> dict[str, object]:
        """Compare WT with one mutant in which every lysine is replaced by arginine."""
        t1 = self.T1(sequence, protein_id) if sequence is not None else None
        up1 = self.UP1()
        manifest = self.T2_all_lysine_to_arginine()
        screen = self.S1 if self.stage_suffix == 'basic' else self.S1_basic
        s1_metrics, _s1_conserved = screen(
            manifest,
            predict_structures=predict_structures,
        )
        r1 = self.R1(manifest, s1_metrics)
        # Score the designated mutant even when it fails the structural gate.
        ub2 = run_ub2(self.ubi_predictor, manifest, self.paths)
        comparison = build_lysine_free_comparison(
            up1=up1,
            ub2=ub2,
            manifest=manifest,
            s1_metrics=s1_metrics,
            paths=self.paths,
        )
        return {
            'T1': t1,
            'UP1': up1,
            'T2_all_lysine_to_arginine': manifest,
            'S1_metrics': s1_metrics,
            'R1': r1,
            'UB2': ub2,
            'lysine_free_comparison': comparison,
        }
