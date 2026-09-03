from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence
import os


DEFAULT_ESM2_MODEL = 'facebook/esm2_t33_650M_UR50D'


def _default_eup_repo() -> Path:
    if Path('/content').exists():
        return Path('/content/external/EUP')
    return Path.home() / '.cache' / 'therapeutic_optimization' / 'EUP'


def _default_hf_cache() -> Path:
    if Path('/content').exists():
        return Path('/content/huggingface')
    return Path.home() / '.cache' / 'huggingface'


@dataclass(slots=True)
class MutationConfig:
    """Hyperparameters controlling T2 mutant generation."""

    threshold: float = 0.40
    mode: str = 'single'  # single | combinatorial
    replacement_aas: tuple[str, ...] = ('A',)
    max_combination_order: int = 2
    max_variants: int = 5000

    def validate(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError('Mutation threshold must be between 0 and 1.')
        if self.mode not in {'single', 'combinatorial'}:
            raise ValueError("Mutation mode must be 'single' or 'combinatorial'.")
        if not self.replacement_aas:
            raise ValueError('At least one replacement amino acid is required.')
        allowed = set('ACDEFGHIKLMNPQRSTVWY')
        normalized = tuple(aa.upper() for aa in self.replacement_aas)
        invalid = sorted(set(normalized) - allowed)
        if invalid:
            raise ValueError(f'Unsupported replacement amino acids: {invalid}')
        if 'K' in normalized:
            raise ValueError("Replacement amino acids should not include 'K'; that would not remove the lysine.")
        if self.max_combination_order < 1:
            raise ValueError('max_combination_order must be at least 1.')
        if self.max_variants < 1:
            raise ValueError('max_variants must be at least 1.')


@dataclass(slots=True)
class StructuralThresholds:
    """
    Transparent, user-adjustable heuristic gates for S1.

    These are workflow defaults, not universal biological cutoffs.
    """

    global_ca_rmsd_max: float = 1.0
    binder_start: int | None = None
    binder_end: int | None = None
    binder_ca_rmsd_max: float = 1.0
    core_ca_rmsd_max: float = 1.0
    local_mean_ca_displacement_max: float = 1.5
    mutation_ca_displacement_max: float = 2.0
    contact_change_fraction_max: float = 0.10
    min_mean_plddt: float = 70.0

    def binder_positions(self) -> tuple[int, int] | None:
        """Return the inclusive binder range, or None when domain-aware analysis is disabled."""
        if self.binder_start is None and self.binder_end is None:
            return None
        if self.binder_start is None or self.binder_end is None:
            raise ValueError('binder_start and binder_end must be set together.')
        if self.binder_start < 1 or self.binder_end < self.binder_start:
            raise ValueError('Binder positions must define a positive, inclusive residue range.')
        return self.binder_start, self.binder_end


@dataclass(slots=True)
class PredictorConfig:
    name: str = 'eup'
    threshold: float = 0.40
    eup_repo_dir: Path = field(default_factory=_default_eup_repo)
    model_cache_dir: Path = field(default_factory=_default_hf_cache)
    force_clone_eup: bool = False


@dataclass(slots=True)
class StructurePredictorConfig:
    name: str = 'colabfold'
    executable: str = 'colabfold_batch'
    extra_args: tuple[str, ...] = ()


@dataclass(slots=True)
class ESM2AnalysisConfig:
    """Runtime and ranking choices for WT-versus-mutant ESM-2 analysis.

    The default 650M model is practical on common Colab GPUs. The larger
    ``facebook/esm2_t36_3B_UR50D`` model can be selected when memory permits.
    """

    model_name: str = DEFAULT_ESM2_MODEL
    mask_batch_size: int = 4
    device: str | None = None
    dtype: Literal['auto', 'float32', 'float16', 'bfloat16'] = 'auto'
    model_cache_dir: str | Path | None = field(default_factory=_default_hf_cache)
    save_per_residue: bool = True
    fail_fast: bool = False
    perplexity_weight: float = 0.70
    representation_weight: float = 0.30

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError('ESM-2 model_name cannot be empty.')
        if self.mask_batch_size < 1:
            raise ValueError('ESM-2 mask_batch_size must be at least 1.')
        if self.dtype not in {'auto', 'float32', 'float16', 'bfloat16'}:
            raise ValueError(f'Unsupported ESM-2 dtype: {self.dtype!r}.')
        if self.perplexity_weight < 0 or self.representation_weight < 0:
            raise ValueError('ESM-2 scoring weights cannot be negative.')
        if self.perplexity_weight + self.representation_weight <= 0:
            raise ValueError('At least one ESM-2 scoring weight must be positive.')


@dataclass(slots=True)
class WorkflowConfig:
    mutation: MutationConfig = field(default_factory=MutationConfig)
    ubiquitination: PredictorConfig = field(default_factory=PredictorConfig)
    structure: StructurePredictorConfig = field(default_factory=StructurePredictorConfig)
    structural_thresholds: StructuralThresholds = field(default_factory=StructuralThresholds)
    esm2: ESM2AnalysisConfig = field(default_factory=ESM2AnalysisConfig)


@dataclass(slots=True)
class ProjectPaths:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> 'ProjectPaths':
        return cls(Path(root).expanduser().resolve())

    @property
    def storage(self) -> Path:
        return self.root / 'storage'

    @property
    def inputs(self) -> Path:
        return self.storage / 'inputs'

    @property
    def wt_fasta(self) -> Path:
        return self.inputs / 'wt_input.fasta'

    @property
    def input_metadata(self) -> Path:
        return self.inputs / 'input_metadata.json'

    @property
    def mutant_fastas(self) -> Path:
        return self.storage / 'mutants' / 'fastas'

    @property
    def ubi_wt(self) -> Path:
        return self.storage / 'ubiquitination' / 'wt'

    @property
    def ubi_mutants(self) -> Path:
        return self.storage / 'ubiquitination' / 'mutants'

    @property
    def structures_wt(self) -> Path:
        return self.storage / 'structures' / 'wt'

    @property
    def structures_mutants(self) -> Path:
        return self.storage / 'structures' / 'mutants'

    @property
    def per_residue(self) -> Path:
        return self.storage / 'structural' / 'per_residue'

    @property
    def esm2(self) -> Path:
        return self.storage / 'esm2'

    @property
    def esm2_per_residue(self) -> Path:
        return self.esm2 / 'per_residue'

    @property
    def figures(self) -> Path:
        return self.storage / 'structural' / 'figures'

    @property
    def tables(self) -> Path:
        return self.storage / 'tables'

    @property
    def logs(self) -> Path:
        return self.storage / 'logs'

    def ensure(self) -> None:
        for path in (
            self.inputs,
            self.mutant_fastas,
            self.ubi_wt,
            self.ubi_mutants,
            self.structures_wt,
            self.structures_mutants,
            self.per_residue,
            self.esm2_per_residue,
            self.figures,
            self.tables,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
