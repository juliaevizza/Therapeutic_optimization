from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..io import read_single_fasta

@dataclass(frozen=True)
class StructurePrediction:
    variant_id: str
    fasta_path: Path
    output_dir: Path


