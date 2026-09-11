from pathlib import Path
from abc import ABC, abstractmethod


#TODO define class for ubiquitination predictor
class Site_Predictor(ABC):
    """
    Predictor interface.
    """
    name_of_predictor: str
    threshold: float
    seq: str
    model: None

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

    #TODO do assert QC on all transformations and uo
    def assert_QC(self) -> bool:
        """
        This will be called by the confirguration file to ensure all the
        information is moving through the pipeline properly.
        """
        assert ()


    #TODO implement
    def read_single_fasta(fasta):
        """
        Open and parse the fasta file into the protein name and the
        sequence.
        """
        seq = None #parse 
        return seq


    #TODO finish threading into this module, moved from lysine free generation 
    def _prediction_summary(predictions: pd.DataFrame, ubi_threshold: float) -> dict[str, float | int]:
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
        return {
            'lysine_count': int(len(predictions)),
             'positive_site_count': int(len(positive)),
            'residues of interest': positive}

