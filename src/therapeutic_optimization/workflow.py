from __future__ import annotations

from pathlib import Path

import pandas as pd

from .parallel import sub module 


class OptimizationWorkflow:
    """Top-level orchestration with explicit T1/UP1/T2/ESM2/S1/R1/UB2/R2 stages."""

    def __init__(
        self,
        project_root: str | Path,
        config: WorkflowConfig | None = None,
    ) -> None:
        self. #TODO initialize modules 
        
#TODO build all modules 
    @property
    def ubi_predictor(self):

#TODO define 
    def T1_basic(self, sequence: str, protein_id: str = 'WT') -> dict:
        self._complex_results = None
        return prepare_wt_input(sequence, protein_id, self.paths)

    

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
