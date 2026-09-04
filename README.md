# Therapeutic Optimization

A modular protein optimization workflow for identifying predicted ubiquitination sites, generating lysine-removal mutants, filtering those mutants for structural preservation, re-running ubiquitination prediction, and ranking the survivors.

The repository is intentionally organized so the notebook is a **lightweight user interface** while the heavy lifting lives in the installable Python package under `src/therapeutic_optimization/`.

## Pipeline logic

```text

### Important input correction

EUP is a **sequence-based** ubiquitination-site predictor. It receives a protein FASTA, not a predicted 3D structure. The structural branch separately sends the WT and mutant FASTAs to ColabFold/AlphaFold.

## Repository layout

```text
therapeutic_optimization/
├── notebooks/
│   └── therapeutic_optimization_colab.ipynb
├── scripts/
│   ├── check_environment.py
│   └── setup_colab.sh
├── src/
│   └── therapeutic_optimization/
│       ├── config.py
│       ├── io.py
│       ├── workflow.py
│       ├── cli.py
│       ├── transformations/
│       │   ├── t1.py
│       │   └── mutants.py
│       ├── ubiquitination_prediction/
│       │   ├── base.py
│       │   ├── eup.py
│       │   └── pipeline.py
│       ├── structural_analysis/
│       │   ├── colabfold.py
│       │   ├── metrics.py
│       │   └── pipeline.py
│       └── ranking/
│           └── pipeline.py
├── storage/
│   ├── inputs/
│   ├── mutants/fastas/
│   ├── ubiquitination/{wt,mutants}/
│   ├── structures/{wt,mutants}/
│   ├── structural/{per_residue,figures}/
│   ├── tables/
│   └── logs/
├── tests/
├── pyproject.toml
└── .gitignore
```

## Stage contracts

### T1 — user input → WT FASTA

`OptimizationWorkflow.T1(sequence, protein_id)` validates the amino-acid alphabet, removes whitespace, writes the canonical WT FASTA, and records input metadata.

**Primary output:** `storage/inputs/wt_input.fasta`

### UP1 — WT ubiquitination prediction

The predictor interface is defined in `ubiquitination_prediction/base.py`. The first implementation is EUP using ESM2-3B residue embeddings and the EUP linear checkpoint.

Every predictor adapter must return the same columns:

```text
variant_id
protein_id
predictor
lysine_position
site
sequence_context
probability
threshold
is_positive
```

This makes T2 and R2 predictor-independent.

**Primary output:** `storage/tables/UP1_wt_ubiquitination.csv`

### T2 — mutant generation

T2 reads UP1, selects lysines above the configured probability threshold, and generates FASTAs.

Two modes already exist:

- `single`: mutate one selected lysine at a time.
- `combinatorial`: generate mutation orders 1 through `max_combination_order`, including all configured replacement amino-acid combinations.

The defaults are deliberately conservative:

```python
MutationConfig(
    threshold=0.40,
    mode="single",
    replacement_aas=("A",),
    max_combination_order=2,
    max_variants=5000,
)
```

The `max_variants` guard prevents accidental combinatorial explosions before expensive structure predictions begin.

**Primary outputs:**

- `storage/mutants/fastas/<variant_id>.fasta`
- `storage/tables/T2_mutation_manifest.csv`

Variant IDs are deterministic. Examples:

```text
K16A
K16A__K43A
K16R__K43A
```

### S1 — structure prediction + preservation screen

The structure adapter is isolated from the metric code. `ColabFoldPredictor` currently wraps `colabfold_batch`; a future AlphaFold server/local adapter can be added without touching the metrics or ranking stages.

Structural metrics include:

- global C-alpha RMSD after alignment
- mean/median/max per-residue C-alpha displacement
- mutation-site displacement
- local displacement around all mutated positions
- radius-of-gyration change
- WT and mutant mean pLDDT from C-alpha B-factors
- global contacts lost/gained/retained
- global contact-change fraction
- local contacts lost/gained at the mutated positions
- mutation-site distance from protein centroid
- residue-volume change
- per-residue displacement CSV and plot

S1 produces both **all structural metrics** and a filtered **structurally conserved** table.

The default conservation gates are workflow heuristics, not universal biological truths:

```python
StructuralThresholds(
    global_ca_rmsd_max=1.0,
    local_mean_ca_displacement_max=1.5,
    mutation_ca_displacement_max=2.0,
    contact_change_fraction_max=0.10,
    min_mean_plddt=70.0,
)
```

Keep these as top-level hyperparameters and tune them against proteins/controls for your actual use case.

**Primary outputs:**

- `storage/tables/S1_structural_metrics.csv`
- `storage/tables/S1_structurally_conserved.csv`
- `storage/structural/per_residue/*.csv`
- `storage/structural/figures/*.png`
- predicted structures under `storage/structures/`

### R1 — structural dropout accounting

R1 merges T2 with S1 and explicitly marks:

```text
ADVANCE_TO_UB2
DROPPED_STRUCTURAL
```

This lets you report how many candidates were lost because they failed the structural screen instead of silently dropping them.

**Primary output:** `storage/tables/R1_structural_screen.csv`

### UB2 — repeat ubiquitination prediction on S1 survivors

The same predictor object used for UP1 is reused for mutant inference. For EUP this means the ESM2 model and classifier can remain loaded in GPU memory rather than being reloaded for each mutant.

Only S1 survivors are sent to UB2.

**Primary output:** `storage/tables/UB2_mutant_ubiquitination.csv`

Per-mutant predictor tables are also written beneath `storage/ubiquitination/mutants/<variant_id>/`.

### R2 — final optimization groups

R2 compares UP1 with UB2 and calculates, for every structurally conserved mutant:

- number of positive WT sites
- number of positive mutant sites
- sum of WT probabilities at intentionally targeted sites
- WT positive-probability burden
- mutant positive-probability burden
- absolute and fractional burden reduction
- WT positive sites removed
- WT positive sites still present
- newly positive lysines that crossed the threshold in the mutant
- structural preservation score

The final groups are:

```text
optimized
needs_further_optimization
```

`optimized` means no lysine in the mutant remains above the configured ubiquitination threshold. Optimized mutants are ranked primarily by how much predicted ubiquitination burden was eliminated, then by structural preservation. Mutants that still contain positive sites are ranked by residual burden/new-site behavior so they can feed a later optimization cycle.

## Runtime dependencies versus scientific outputs

Do **not** commit third-party model weights or the Hugging Face model cache to GitHub.

Keep these as runtime assets:

- EUP source repository and Git LFS checkpoint
- ESM2-3B model cache
- ColabFold/JAX installation
- any large ColabFold databases if you later move to a local database workflow

Keep these as pipeline outputs:

- input WT FASTA
- mutant FASTAs
- UP1 / T2 / S1 / R1 / UB2 / R2 CSVs
- rank-1 WT and mutant structures
- per-residue structural metrics
- structural plots
- logs needed to audit a run

The provided `.gitignore` excludes generated storage outputs by default. If you want a particular run to be a versioned benchmark, copy that run into a deliberately named `examples/` or `benchmarks/` directory rather than committing the live `storage/` workspace.

## EUP runtime notes

The current EUP adapter uses:

- `facebook/esm2_t36_3B_UR50D`
- 2,560-dimensional residue embeddings
- the EUP `DNNLinearModel` checkpoint
- default probability threshold `0.40`
- maximum full-sequence length `1022` residues

The EUP repository is cloned to `/content/external/EUP` by default in Colab. It is intentionally **not** cloned into mounted Google Drive because Git LFS hooks can be unreliable there.

The ESM2 cache defaults to `/content/huggingface` in Colab so the very large model is not written into your project repository.

## ColabFold runtime notes

ColabFold is a separate optional system dependency because the correct JAX/CUDA package depends on the runtime GPU. Install the build recommended by the current ColabFold documentation, then verify:

```bash
python scripts/check_environment.py
```

The pipeline expects `colabfold_batch` on `PATH` unless you override `StructurePredictorConfig.executable`.

## Typical notebook use

The notebook exposes the important hyperparameters near the top:

```python
UBI_THRESHOLD = 0.40
MUTATION_MODE = "single"
REPLACEMENT_AAS = ("A",)
MAX_COMBINATION_ORDER = 2
MAX_VARIANTS = 5000

GLOBAL_CA_RMSD_MAX = 1.0
LOCAL_MEAN_CA_DISPLACEMENT_MAX = 1.5
MUTATION_CA_DISPLACEMENT_MAX = 2.0
CONTACT_CHANGE_FRACTION_MAX = 0.10
MIN_MEAN_PLDDT = 70.0
```

Then it executes the stages visibly:

```python
t1 = workflow.T1(WT_SEQUENCE, PROTEIN_ID)
up1 = workflow.UP1()
t2 = workflow.T2(up1)
s1_metrics, s1_conserved = workflow.S1(t2)
r1 = workflow.R1(t2, s1_metrics)
ub2 = workflow.UB2(s1_conserved)
r2_all, optimized, needs_more = workflow.R2(up1, ub2, s1_conserved)
```

## CLI

After installation:

```bash
pip install -e '.[eup]'
therapeutic-optimize --stage T1 --sequence "MSEQUENCE..." --protein-id my_protein
therapeutic-optimize --stage UP1
therapeutic-optimize --stage T2
therapeutic-optimize --stage S1
therapeutic-optimize --stage R1
therapeutic-optimize --stage UB2
therapeutic-optimize --stage R2
```

For existing ColabFold outputs:

```bash
therapeutic-optimize --stage S1 --analyze-only
```

## Adding another ubiquitination predictor

1. Create an adapter subclassing `UbiquitinationPredictor`.
2. Return the standard predictor columns.
3. Register it in `ubiquitination_prediction/pipeline.py::build_predictor`.
4. Set `PredictorConfig(name="your_predictor")`.

T2, UB2, and R2 require no changes.

## Adding another structure predictor

1. Create an adapter exposing `predict(fasta_path, output_dir) -> structure_path`.
2. Register it in `structural_analysis/pipeline.py::build_structure_predictor`.
3. Keep the rank-1 output as PDB/CIF so `metrics.py` can consume it.

## Development tests

The unit tests intentionally avoid GPU/network calls and test the workflow logic independently of EUP/ColabFold:

```bash
pip install -e '.[dev]'
pytest -q
```

Before publishing this project, add an explicit open-source license and replace the placeholder GitHub URL in the Colab notebook.
