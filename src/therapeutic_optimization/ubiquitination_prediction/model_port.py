from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


#TODO look over
class UbiquitinationPredictor(ABC):
    """Predictor interface so EUP can be replaced without changing T2/R2."""

    name: str
    threshold: float

    @abstractmethod
    def predict_sequence(
        self,
        sequence: str,
        protein_id: str,
        variant_id: str,
        output_dir: Path | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError
