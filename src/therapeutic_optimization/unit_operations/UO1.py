from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
from pandas import DataFrame as df
from Bio import SeqIO



class Site_Predictor(ABC):
    """
    Predictor interface.
    """
    name_of_predictor: str
    threshold: float
    seq: str
    model: None


    #module specific methods.
    @abstractmethod
    def build_predictor(self):
        """
        This function will build the predictor and store in in a variable.
        """
        pass

    @abstractmethod
    def predict_sites(self, sequence, output_dir):
        """
        This function will run the predictor.
        """
        pass


    #general methods
    def assert_QC(results) -> bool:
        """
        This will be called by the confirguration file to ensure all the
        information is moving through the pipeline properly.
        """
        assert(type(results) == pd.Dataframe)
        assert(df.results["site"].empty) == False

    def _prediction_summary(predictions: pd.DataFrame, ubi_threshold: float) -> list:
        ""
        "Returns data from ubiquitnatin prediciton."
        ""
        if predictions.empty:
           return {
            'lysine_count': 0,
            'positive_site_count': 0,
        }
        probabilities = predictions['probability'].astype(float)
        positive = probabilities > ubi_threshold
        results = []
        for r in positive: 
            residue = "K" + str(r)
            results.append(residue)
        return results
