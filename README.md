# Therapeutic Optimization

This modular protein optimization tool chain for identifying predicted ubiquitination sites, generating lysine-removal mutants, filtering those mutants for structural preservation, re-running ubiquitination prediction, and ranking the survivors.

The repository is intentionally organized so the notebook is a **lightweight user interface** while the heavy lifting lives in the installable Python package under `src/therapeutic_optimization/`.

For use, simple open the google colab attached, connect your runtime to a GPU, and click run all. You will first be prompted for the wild type sequence and hyper parameters before imports, so that you may click run and step away while all the computing occurs. There is an option in the code to save the result to drive that is automatically true. If you are testing the code for and don't want results accumulating in your drive, control+F "SAVE_TO_DRIVE" and set it to false. 

 <a href="https://colab.research.google.com/github/juliaevizza/Therapeutic_optimization/blob/main/notebooks/therapeutic_optimization_colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in colab!" height="40">
</a>

## Toolchain logic

