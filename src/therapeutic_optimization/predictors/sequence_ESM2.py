"""
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#TODO: resolve
import esm
import torch


REQUIRED_MANIFEST_COLUMNS = {"protein_id", "fasta_path"}

SUMMARY_COLUMNS = 



class ESM2Scorer:
    """Lazily load ESM-2 and calculate pseudo-perplexity/representations."""

    def __init__(self) -> None:
        self.config = ESM2AnalysisConfig()
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

def _build_comparison(variant_id: str, wt: SequenceScore, mutant: SequenceScore,
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

    pooled_cosine = _cosine_similarity(
        wt.pooled_representation,
        mutant.pooled_representation,
    )

    summary = {
        "variant_id": variant_id,
        "model_name": model_name,

        "delta_pseudo_log_likelihood": (
            mutant.pseudo_log_likelihood - wt.pseudo_log_likelihood
        ),

        "delta_mean_log_probability": (
            mutant.mean_log_probability - wt.mean_log_probability
        ),

        "pooled_representation_cosine_distance": 1.0 - pooled_cosine,
        "pooled_representation_l2_distance": float(
            np.linalg.norm(mutant.pooled_representation - wt.pooled_representation)
        ),
        
        
        "per_residue_path": str(per_residue_path) if per_residue_path else None,
        "analysis_status": "PASS",
        "analysis_error": None,
    }
    return summary

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
    manifest_path = args.manifest or paths.table("T2_mutation_manifest.csv")
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
    print(args.output or paths.table("ESM2_mutant_comparison.csv"))





################_________________________________


def sort_mutant_sequences(protein_manifest): 
    for i in protein_manifest:
        sequence = protein_manifest.iloc[i]["sequence"]
        ## either skip wt or be prepared to handle it later
        score_mut_sequence(sequence)
        #sort dataframe from score highest to lowest

def score_mut_sequence(seq):
    pass
