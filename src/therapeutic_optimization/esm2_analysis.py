"""Compare T2 mutant sequences with WT using an ESM-2 masked language model.

This module is intentionally self-contained so it can be dropped into
``src/therapeutic_optimization/`` without changing the rest of the package.

Typical use after T2::

    from therapeutic_optimization.esm2_analysis import run_esm2_analysis

    t2 = workflow.T2(up1)
    esm2_results = run_esm2_analysis(t2, workflow.paths)

The primary output is ``storage/tables/ESM2_mutant_comparison.csv``.  By
default, residue-level representation and log-probability changes are also
written beneath ``storage/esm2/per_residue/``.

ESM-2 is a masked language model, so "perplexity" here means pseudo-perplexity:
each residue is masked in turn and its conditional log-probability is scored.
This costs roughly one model evaluation per sequence residue (evaluations are
batched) but gives a meaningful, directly comparable score for equal-length
substitution mutants.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_ESM2_MODEL, ESM2AnalysisConfig, ProjectPaths
from .io import normalize_sequence, parse_mutation, read_single_fasta


REQUIRED_MANIFEST_COLUMNS = {"variant_id", "fasta_path"}

SUMMARY_COLUMNS = [
    "variant_id",
    "mutation_spec",
    "model_name",
    "sequence_length",
    "mutation_count",
    "wt_pseudo_log_likelihood",
    "mutant_pseudo_log_likelihood",
    "delta_pseudo_log_likelihood",
    "wt_mean_log_probability",
    "mutant_mean_log_probability",
    "delta_mean_log_probability",
    "wt_pseudo_perplexity",
    "mutant_pseudo_perplexity",
    "delta_pseudo_perplexity",
    "pseudo_perplexity_percent_change",
    "pooled_representation_cosine_similarity",
    "pooled_representation_cosine_distance",
    "pooled_representation_l2_distance",
    "mean_residue_representation_cosine_similarity",
    "mean_residue_representation_cosine_distance",
    "mean_residue_representation_l2_distance",
    "max_residue_representation_l2_distance",
    "mutation_site_representation_cosine_similarity",
    "mutation_site_representation_cosine_distance",
    "mutation_site_representation_l2_distance",
    "per_residue_path",
    "analysis_status",
    "analysis_error",
]


@dataclass(slots=True)
class SequenceScore:
    """ESM-2 outputs retained for one sequence comparison."""

    sequence: str
    residue_log_probabilities: np.ndarray
    residue_representations: np.ndarray
    pooled_representation: np.ndarray

    @property
    def pseudo_log_likelihood(self) -> float:
        return float(self.residue_log_probabilities.sum())

    @property
    def mean_log_probability(self) -> float:
        return float(self.residue_log_probabilities.mean())

    @property
    def pseudo_perplexity(self) -> float:
        # The upper bound prevents OverflowError while preserving an explicit
        # infinite score for an extremely improbable sequence.
        exponent = -self.mean_log_probability
        return math.exp(exponent) if exponent < 709.0 else math.inf


class ESM2Scorer:
    """Lazily load ESM-2 and calculate pseudo-perplexity/representations."""

    def __init__(self, config: ESM2AnalysisConfig | None = None) -> None:
        self.config = config or ESM2AnalysisConfig()
        self.config.validate()
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ESM-2 dependencies are missing. Install the project's optional "
                "EUP dependencies, e.g. `pip install -e '.[eup]'`."
            ) from exc

        if self.config.device is not None:
            device = torch.device(self.config.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        dtype_name = self.config.dtype
        if dtype_name == "auto":
            dtype_name = "float16" if device.type == "cuda" else "float32"
        if device.type == "cpu" and dtype_name == "float16":
            raise ValueError("float16 ESM-2 inference is not supported on CPU; use float32.")
        dtype = getattr(torch, dtype_name)

        cache_dir = (
            str(Path(self.config.model_cache_dir).expanduser())
            if self.config.model_cache_dir is not None
            else None
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir,
        )
        if tokenizer.mask_token_id is None:
            raise ValueError(f"Tokenizer for {self.config.model_name!r} has no mask token.")
        model = AutoModelForMaskedLM.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _encode(self, sequence: str) -> tuple[Any, Any, Any]:
        tokenizer = self._tokenizer
        encoded = tokenizer(
            sequence,
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special_tokens_mask = encoded.pop("special_tokens_mask")[0]
        residue_token_indices = self._torch.nonzero(
            special_tokens_mask == 0,
            as_tuple=False,
        ).flatten()
        if len(residue_token_indices) != len(sequence):
            raise RuntimeError(
                "ESM-2 tokenizer mapping failed: expected "
                f"{len(sequence)} residue tokens, found {len(residue_token_indices)}."
            )

        max_positions = getattr(self._model.config, "max_position_embeddings", None)
        if max_positions is not None and encoded["input_ids"].shape[1] > max_positions:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds the {self.config.model_name} "
                f"context limit ({max_positions} tokens including special tokens)."
            )
        return encoded["input_ids"], encoded["attention_mask"], residue_token_indices

    def score_sequence(self, sequence: str) -> SequenceScore:
        """Score one sequence without truncation.

        Pseudo-log-likelihood is calculated by masking every residue exactly
        once.  Representations are final-layer embeddings from one unmasked
        forward pass, restricted to residue tokens.
        """

        sequence = normalize_sequence(sequence)
        self._load_model()
        torch = self._torch
        model = self._model
        device = self._device
        input_ids, attention_mask, residue_indices = self._encode(sequence)

        device_ids = input_ids.to(device)
        device_attention = attention_mask.to(device)
        device_residue_indices = residue_indices.to(device)

        with torch.inference_mode():
            base_outputs = model.base_model(
                input_ids=device_ids,
                attention_mask=device_attention,
                return_dict=True,
            )
            residue_representations = (
                base_outputs.last_hidden_state[0, device_residue_indices]
                .float()
                .cpu()
                .numpy()
            )

        log_probability_chunks: list[np.ndarray] = []
        batch_size = self.config.mask_batch_size
        for start in range(0, len(sequence), batch_size):
            positions = residue_indices[start : start + batch_size]
            current_size = len(positions)
            masked_ids = input_ids.repeat(current_size, 1)
            batch_rows = torch.arange(current_size)
            masked_ids[batch_rows, positions] = self._tokenizer.mask_token_id
            batch_attention = attention_mask.repeat(current_size, 1)
            targets = input_ids[0, positions]

            with torch.inference_mode():
                logits = model(
                    input_ids=masked_ids.to(device),
                    attention_mask=batch_attention.to(device),
                    return_dict=True,
                ).logits
                selected_logits = logits[
                    torch.arange(current_size, device=device),
                    positions.to(device),
                ]
                log_probabilities = torch.log_softmax(selected_logits.float(), dim=-1)
                true_log_probabilities = log_probabilities.gather(
                    1,
                    targets.to(device).unsqueeze(1),
                ).squeeze(1)
            log_probability_chunks.append(true_log_probabilities.cpu().numpy())

        residue_log_probabilities = np.concatenate(log_probability_chunks)
        return SequenceScore(
            sequence=sequence,
            residue_log_probabilities=residue_log_probabilities,
            residue_representations=residue_representations,
            pooled_representation=residue_representations.mean(axis=0),
        )

    def release(self) -> None:
        """Release model references and clear CUDA cache when applicable."""

        torch = self._torch
        device = self._device
        self._tokenizer = None
        self._model = None
        if torch is not None and device is not None and device.type == "cuda":
            torch.cuda.empty_cache()


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return math.nan
    return float(np.dot(left, right) / denominator)


def _rowwise_representation_metrics(
    wt_representations: np.ndarray,
    mutant_representations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dot_products = np.sum(wt_representations * mutant_representations, axis=1)
    denominators = np.linalg.norm(wt_representations, axis=1) * np.linalg.norm(
        mutant_representations,
        axis=1,
    )
    cosine_similarities = np.divide(
        dot_products,
        denominators,
        out=np.full(dot_products.shape, np.nan, dtype=float),
        where=denominators != 0,
    )
    l2_distances = np.linalg.norm(mutant_representations - wt_representations, axis=1)
    return cosine_similarities, l2_distances


def _mutation_positions(
    wt_sequence: str,
    mutant_sequence: str,
    mutation_spec: str,
) -> list[int]:
    changed = [
        position
        for position, (wt_aa, mutant_aa) in enumerate(
            zip(wt_sequence, mutant_sequence),
            start=1,
        )
        if wt_aa != mutant_aa
    ]
    if not mutation_spec or mutation_spec.lower() == "nan":
        return changed

    declared: list[int] = []
    for mutation in mutation_spec.split(";"):
        _wt_aa, position, _mutant_aa = parse_mutation(mutation)
        declared.append(position)
    if sorted(set(declared)) != changed:
        raise ValueError(
            f"mutation_spec positions {sorted(set(declared))} do not match actual "
            f"WT/mutant differences {changed}."
        )
    return changed


def _build_comparison(
    variant_id: str,
    mutation_spec: str,
    wt: SequenceScore,
    mutant: SequenceScore,
    model_name: str,
    per_residue_path: Path | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(wt.sequence) != len(mutant.sequence):
        raise ValueError(
            f"{variant_id}: WT and mutant lengths differ "
            f"({len(wt.sequence)} versus {len(mutant.sequence)}); residue-aligned "
            "representation comparison requires substitution-only mutants."
        )
    mutation_positions = _mutation_positions(wt.sequence, mutant.sequence, mutation_spec)
    if not mutation_positions:
        raise ValueError(f"{variant_id}: mutant sequence is identical to WT.")

    residue_cosine, residue_l2 = _rowwise_representation_metrics(
        wt.residue_representations,
        mutant.residue_representations,
    )
    mutation_indices = np.asarray(mutation_positions, dtype=int) - 1
    pooled_cosine = _cosine_similarity(
        wt.pooled_representation,
        mutant.pooled_representation,
    )

    wt_pp = wt.pseudo_perplexity
    mutant_pp = mutant.pseudo_perplexity
    pp_percent_change = (
        100.0 * (mutant_pp - wt_pp) / wt_pp if wt_pp != 0.0 else math.nan
    )
    summary = {
        "variant_id": variant_id,
        "mutation_spec": mutation_spec,
        "model_name": model_name,
        "sequence_length": len(mutant.sequence),
        "mutation_count": len(mutation_positions),
        "wt_pseudo_log_likelihood": wt.pseudo_log_likelihood,
        "mutant_pseudo_log_likelihood": mutant.pseudo_log_likelihood,
        "delta_pseudo_log_likelihood": (
            mutant.pseudo_log_likelihood - wt.pseudo_log_likelihood
        ),
        "wt_mean_log_probability": wt.mean_log_probability,
        "mutant_mean_log_probability": mutant.mean_log_probability,
        "delta_mean_log_probability": (
            mutant.mean_log_probability - wt.mean_log_probability
        ),
        "wt_pseudo_perplexity": wt_pp,
        "mutant_pseudo_perplexity": mutant_pp,
        "delta_pseudo_perplexity": mutant_pp - wt_pp,
        "pseudo_perplexity_percent_change": pp_percent_change,
        "pooled_representation_cosine_similarity": pooled_cosine,
        "pooled_representation_cosine_distance": 1.0 - pooled_cosine,
        "pooled_representation_l2_distance": float(
            np.linalg.norm(mutant.pooled_representation - wt.pooled_representation)
        ),
        "mean_residue_representation_cosine_similarity": float(
            np.nanmean(residue_cosine)
        ),
        "mean_residue_representation_cosine_distance": float(
            np.nanmean(1.0 - residue_cosine)
        ),
        "mean_residue_representation_l2_distance": float(np.mean(residue_l2)),
        "max_residue_representation_l2_distance": float(np.max(residue_l2)),
        "mutation_site_representation_cosine_similarity": float(
            np.nanmean(residue_cosine[mutation_indices])
        ),
        "mutation_site_representation_cosine_distance": float(
            np.nanmean(1.0 - residue_cosine[mutation_indices])
        ),
        "mutation_site_representation_l2_distance": float(
            np.mean(residue_l2[mutation_indices])
        ),
        "per_residue_path": str(per_residue_path) if per_residue_path else None,
        "analysis_status": "PASS",
        "analysis_error": None,
    }
    per_residue = pd.DataFrame(
        {
            "variant_id": variant_id,
            "position": np.arange(1, len(wt.sequence) + 1),
            "wt_aa": list(wt.sequence),
            "mutant_aa": list(mutant.sequence),
            "is_mutation_site": [
                position in set(mutation_positions)
                for position in range(1, len(wt.sequence) + 1)
            ],
            "wt_log_probability": wt.residue_log_probabilities,
            "mutant_log_probability": mutant.residue_log_probabilities,
            "delta_log_probability": (
                mutant.residue_log_probabilities - wt.residue_log_probabilities
            ),
            "representation_cosine_similarity": residue_cosine,
            "representation_cosine_distance": 1.0 - residue_cosine,
            "representation_l2_distance": residue_l2,
        }
    )
    return summary, per_residue


def _failed_record(
    variant_id: str,
    mutation_spec: str,
    model_name: str,
    status: str,
    error: str,
) -> dict[str, Any]:
    record = {column: None for column in SUMMARY_COLUMNS}
    record.update(
        {
            "variant_id": variant_id,
            "mutation_spec": mutation_spec,
            "model_name": model_name,
            "analysis_status": status,
            "analysis_error": error,
        }
    )
    return record


def run_esm2_analysis(
    manifest: pd.DataFrame,
    paths: ProjectPaths,
    config: ESM2AnalysisConfig | None = None,
    *,
    scorer: ESM2Scorer | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Score every valid T2 mutant and compare it directly with WT.

    Positive ``delta_pseudo_log_likelihood`` means ESM-2 considers the mutant
    more probable than WT; negative ``delta_pseudo_perplexity`` has the same
    favorable direction.  Representation distances are descriptive and do not
    have a universal biological pass/fail cutoff.

    ``scorer`` is injectable for testing.  When supplied, its config determines
    the model label and the explicit ``config`` argument must be omitted.
    """

    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"T2 manifest is missing required columns: {sorted(missing)}")
    if scorer is not None and config is not None:
        raise ValueError("Pass either config or scorer, not both.")

    paths.ensure()
    output = Path(output_path) if output_path is not None else (
        paths.tables / "ESM2_mutant_comparison.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    if manifest.empty:
        empty = pd.DataFrame(columns=SUMMARY_COLUMNS)
        empty.to_csv(output, index=False)
        return empty

    active_scorer = scorer or ESM2Scorer(config)
    active_config = active_scorer.config
    active_config.validate()
    _wt_id, wt_sequence = read_single_fasta(paths.wt_fasta)
    wt_score = active_scorer.score_sequence(wt_sequence)

    per_residue_dir = paths.esm2_per_residue
    if active_config.save_per_residue:
        per_residue_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        variant_id = str(row["variant_id"])
        mutation_spec = str(row.get("mutation_spec", ""))
        manifest_status = str(row.get("status", "PASS"))
        if manifest_status != "PASS":
            records.append(
                _failed_record(
                    variant_id,
                    mutation_spec,
                    active_config.model_name,
                    "SKIPPED",
                    f"T2 status is {manifest_status}: {row.get('error', '')}",
                )
            )
            continue

        try:
            _mutant_id, mutant_sequence = read_single_fasta(row["fasta_path"])
            mutant_score = active_scorer.score_sequence(mutant_sequence)
            per_residue_path = (
                per_residue_dir / f"{variant_id}_esm2.csv"
                if active_config.save_per_residue
                else None
            )
            summary, per_residue = _build_comparison(
                variant_id=variant_id,
                mutation_spec=mutation_spec,
                wt=wt_score,
                mutant=mutant_score,
                model_name=active_config.model_name,
                per_residue_path=per_residue_path,
            )
            if per_residue_path is not None:
                per_residue.to_csv(per_residue_path, index=False)
            records.append(summary)
        except Exception as exc:
            if active_config.fail_fast:
                raise
            records.append(
                _failed_record(
                    variant_id,
                    mutation_spec,
                    active_config.model_name,
                    "FAILED",
                    str(exc),
                )
            )

    result = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    """Command-line entry point for running this drop-in module directly."""

    parser = argparse.ArgumentParser(
        description="Compare T2 mutants with WT using ESM-2 pseudo-perplexity and embeddings."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-name", default=DEFAULT_ESM2_MODEL)
    parser.add_argument("--mask-batch-size", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--no-per-residue", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.project_root)
    manifest_path = args.manifest or paths.tables / "T2_mutation_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    result = run_esm2_analysis(
        manifest,
        paths,
        ESM2AnalysisConfig(
            model_name=args.model_name,
            mask_batch_size=args.mask_batch_size,
            device=args.device,
            dtype=args.dtype,
            model_cache_dir=args.model_cache_dir,
            save_per_residue=not args.no_per_residue,
            fail_fast=args.fail_fast,
        ),
        output_path=args.output,
    )
    pass_count = int(result["analysis_status"].eq("PASS").sum())
    print(f"ESM-2 analysis complete: {pass_count}/{len(result)} mutants passed.")
    print(args.output or paths.tables / "ESM2_mutant_comparison.csv")


if __name__ == "__main__":
    main()
