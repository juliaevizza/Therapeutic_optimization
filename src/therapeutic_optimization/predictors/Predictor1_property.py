from abc import ABC, abstractmethod
import pandas as pd

class SitePredictor(ABC):
    """
    This class is meant to be ammendable to any number of property predictors
    It has initially been built with a ubiquitin predictor, but could more 
    broadly apply to other property predictors so long as it comply with this parent
    class. 
    """
    name_of_predictor: str
    threshold: float
    seq: str
    model: None


    #module specific methods.
    @abstractmethod
    def build_predictor(self):
        """
        This function will build the predictor and store it in a variable.
        """
        pass

    @abstractmethod
    def predict_sites(self, sequence):
        """
        This function will run the predictor once built. It takes the 
        wildtype sequence argument to feed it into the predictor.
        """
        pass


    #general methods
    def assert_QC(results) -> bool:
        """
        This will be called to ensure all the information is 
        moving through the pipeline properly.
        """
        assert(type(results) == pd.DataFrame)
        assert not results["site"].empty

    def _prediction_summary(predictions: pd.DataFrame, ubi_threshold: float) -> list:
        """
        Returns data from ubiquitination predicition as a string list. This string
        list will be passed into the next module to generate the mutants. It takes the
        the predictions data frame (for now... may change to a manifest). And takes
        the ubiqitin threshold as parameter to choose what the cut off for sites is. 
        """
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
