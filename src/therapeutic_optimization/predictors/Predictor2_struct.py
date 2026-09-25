from pathlib import Path
from abc import ABC, abstractmethod


#TODO look over class and how it compares to standard code surronding colabfold/alphafold
class StructuralAnalysis(ABC):
    """
    This class is similar to Predict1's SitePredictor parent class. It offers
    lightweight guidelines for how to integrate different structural scoring metrics into
    this pipeline. It will be compatible with ESM-2 and AlphaFold models to quantify the 
    structural changes. 
    """

    @abstractmethod
    def __init__() -> None:
        pass

    @abstractmethod
    def predict_structure(self, batch: list, portion: tuple, batch_output_dir: Path,) -> dict[str, list[Path]]:
        """
        Takes a list of seqs in list total "batch", queries the structural 
        predictor, and return a list .
        """

    @abstractmethod
    def assertQC():
        """
        This will be what ensures that output is always consistent leaving the structural analysis. 
        """

    @abstractmethod
    def run_metrics():
        """
        This will compare each mutant to the wildtype and give it a corresponding score.
        """

    