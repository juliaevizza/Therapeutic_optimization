from __future__ import annotations

from itertools import combinations, product
from math import prod
from pathlib import Path

import pandas as pd

from ..config import MutationConfig, ProjectPaths
from ..io import apply_mutations, read_single_fasta, write_fasta

REQUIRED_UP1_COLUMNS = {'lysine_position', 'probability'}


#TODO generate all experimentally interesting data structures of mutants 
def generate_mutant() 