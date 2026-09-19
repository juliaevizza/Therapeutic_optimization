from pathlib import Path
import subprocess
from uuid import uuid4

import shutil
#move structural analysis into here
import sys
sys.path.insert(0, "/content")  # Replace with your .py file's folder

from abc import ABC, abstractmethod
from colabfold_runner import fold_one, fold_batch, fold
from Bio import SeqIO

from ..config import ProjectPaths

#look over class and how it compares to standard code surronding colabfold/alphafold
class StructuralAnalysis(ABC):
    """Thin structure-predictor adapter around whichever model I am calling"""

    @abstractmethod
    def __init__() -> None:
        pass

    @abstractmethod
    def predict_structure(self, batch: list, portion: tuple, batch_output_dir: Path,) -> dict[str, list[Path]]:
        """Takes a list of seqs in list "batch", queries the structural predictor, and ."""

    @abstractmethod
    def standardize_analysis():
        """This will be what ensures that output is always consistent leaving the structural analysis. """
        
#metrics