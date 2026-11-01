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
import torch


class ESM2Scorer:
    """Lazily load ESM-2 and calculate pseudo-perplexity/representations."""

    def __init__(self, predictor, config) -> None:

        if predictor._esm_model is None or predictor._tokenizer is None:
            raise RuntimeError("Build the EUP predictor before scoring sequences.")

        if config.mask_batch_size < 1:
            raise ValueError("mask_batch_size must be at least 1.")
    
        self.config = config
        self._torch = predictor._torch
        self._tokenizer = predictor._tokenizer
        self._model = predictor._esm_model
        self._device = predictor._device

        
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

    def score_sequence_residue_log_probabilities(self, sequence: str):
        """Score one sequence without truncation.

        Pseudo-log-likelihood is calculated by masking every residue exactly
        once.  Representations are final-layer embeddings from one unmasked
        forward pass, restricted to residue tokens.
        """
        torch = self._torch
        model = self._model
        device = self._device
        input_ids, attention_mask, residue_indices = self._encode(sequence)

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
        return residue_log_probabilities.mean()


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

################_________________________________

def sort_mutant_sequences(protein_manifest): 
    for i in protein_manifest:
        sequence = protein_manifest.iloc[i]["sequence"]
        ## either skip wt or be prepared to handle it later
        score_mut_sequence(sequence)
        #sort dataframe from score highest to lowest

def score_mut_sequence(seq):
    pass
