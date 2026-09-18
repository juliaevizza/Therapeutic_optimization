from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


import pandas as pd

from ..config import PredictorConfig, ProjectPaths
from ..io import read_single_fasta
from .tUP1_EUP import Site_Predictor

EUP_REPOSITORY_URL = 'https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction.git'
ESM_MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'
ESM_EMBEDDING_DIMENSION = 2560
ESM_MAX_RESIDUES = 1022
EUP_CHECKPOINT = Path('Model/DNNLinerModel/DNNLinermodel_checkpoint_epoch_34.pth')


def _run(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, 
                            stderr=subprocess.STDOUT, check=False,)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )


def _sequence_context(sequence: str, position: int, radius: int = 10) -> str:
    zero = position - 1
    return sequence[max(0, zero - radius): min(len(sequence), zero + radius + 1)]


class EUPPredictor(Site_Predictor):
    """
    EUP adapter using ESM2-3B residue embeddings + the published linear checkpoint.

    Models are loaded lazily and retained in memory, which is important for UB2
    because many mutant FASTAs are scored in one session.
    """

    name = 'EUP'

    def __init__(self, threshold: float = 0.40, eup_repo_dir: str | Path = '/content/external/EUP',
        model_cache_dir: str | Path = '/content/huggingface', ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError('EUP threshold must be between 0 and 1.')
        self.threshold = float(threshold)
        self.eup_repo_dir = Path(eup_repo_dir).expanduser()
        self.model_cache_dir = Path(model_cache_dir).expanduser()
        self._tokenizer = None
        self._esm_model = None
        self._classifier = None
        self._torch = None
        self._device = None

    @property
    def checkpoint_path(self) -> Path:
        return self.eup_repo_dir / EUP_CHECKPOINT

    def release(self) -> None:
        """
        Free EUP GPU models between WT inference and the adaptive search.
        """
        self._tokenizer = None
        self._esm_model = None
        self._classifier = None
        if self._torch is not None and self._device is not None and self._device.type == 'cuda':
            self._torch.cuda.empty_cache()

    def is_valid_git_repo(path: Path) -> bool:
        result = subprocess.run(
            ['git', '-C', str(path), 'rev-parse', '--is-inside-work-tree'],
            capture_output=True,
            text=True,
            check=False,)
        return result.returncode == 0 and result.stdout.strip() == 'true'

    def prepare_repository(self) -> None:
        """
        Mount the EUP repository from github into the a local run time path
        """
        resolved = self.eup_repo_dir.resolve()
        if str(resolved).startswith('/content/drive/'):
            raise ValueError(
                'Do not clone EUP inside mounted Google Drive. '
                'Use /content/external/EUP or another local runtime path.'
            )
            
        if not is_valid_git_repo(self.eup_repo_dir):
            _run(['git', 'clone', EUP_REPOSITORY_URL, str(self.eup_repo_dir)])
        if not self.eup_repo_dir.exists():
            self.eup_repo_dir.parent.mkdir(parents=True, exist_ok=True)
            _run(['git', 'clone', EUP_REPOSITORY_URL, str(self.eup_repo_dir)])

        checkpoint = self.checkpoint_path

        if not checkpoint.exists() or checkpoint.stat().st_size < 1000:
            if shutil.which('git-lfs') or shutil.which('git'):
                try:
                    _run(['git', 'lfs', 'pull'], cwd=self.eup_repo_dir)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f'EUP checkpoint is missing or still a Git LFS pointer: {checkpoint}. '
                        'Install git-lfs and run `git lfs pull` inside the EUP repository.'
                    ) from exc
            raise RuntimeError(f'Valid EUP checkpoint not found at {checkpoint}.')
    


    #TODO figure out torch importing
    def build_predictor(self) -> None:
        """
        This will intialize the model, a very important, kinda odd step
        to think about, but it is an object.
        """
        if self._esm_model is not None:
            return
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModel, AutoTokenizer
            
        except ImportError as exc:
            raise RuntimeError(
                'EUP dependencies are missing. Install torch, transformers, accelerate, and safetensors.'
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                'EUP ESM2-3B inference requires a CUDA GPU in this implementation. '
                'In Colab select a GPU runtime.')

        self.prepare_repository()
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device('cuda')

        tokenizer = AutoTokenizer.from_pretrained(
            ESM_MODEL_NAME,
            cache_dir=self.model_cache_dir,
        )
        esm_model = AutoModel.from_pretrained(
            ESM_MODEL_NAME,
            cache_dir=self.model_cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={'': 0},
        )
        esm_model.eval()

        class DNNLinearModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(ESM_EMBEDDING_DIMENSION, 1)

            def forward(self, features):
                return self.fc1(features)

        try:
            state_dict = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(self.checkpoint_path, map_location='cpu')

        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        cleaned = {
            key.removeprefix('module.'): value
            for key, value in state_dict.items()
        }
        expected = {'fc1.weight', 'fc1.bias'}
        if set(cleaned) != expected:
            raise ValueError(
                f'Unexpected EUP checkpoint keys. Expected {sorted(expected)}, got {sorted(cleaned)}.'
            )
        if tuple(cleaned['fc1.weight'].shape) != (1, ESM_EMBEDDING_DIMENSION):
            raise ValueError('Unexpected EUP classifier weight shape.')

        classifier = DNNLinearModel()
        classifier.load_state_dict(cleaned, strict=True)
        classifier = classifier.to(device=device, dtype=torch.float32)
        classifier.eval()

        self._torch = torch
        self._device = device
        self._tokenizer = tokenizer
        self._esm_model = esm_model
        self._classifier = classifier



    def predict_sites(self, sequence: str) -> pd.DataFrame:
        """
        This will predict the sites of sequence that are being
        ubiquitinated. It wil return it as an entire pdDataFrame
        with the column "site" containing all of the resiudes of 
        interst.
        """
        if len(sequence) > ESM_MAX_RESIDUES:
            raise ValueError(
                f'Sequence length {len(sequence)} exceeds this EUP full-sequence limit of {ESM_MAX_RESIDUES}; '
                'the sequence will not be silently truncated.'
            )

        #determine lysine positions () (example of casting protein to a 1 index)
        lysine_positions = [i for i, aa in enumerate(sequence, start=1) if aa == 'K']

        torch = self._torch
        tokenizer = self._tokenizer
        esm_model = self._esm_model
        classifier = self._classifier
        device = self._device

        encoded = tokenizer(
            sequence,
            return_tensors='pt',
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = encoded.pop('special_tokens_mask')[0]
        residue_token_indices = torch.nonzero(special_tokens_mask == 0,as_tuple=False,).flatten()
        if len(residue_token_indices) != len(sequence):
            raise RuntimeError(
                f'ESM tokenizer residue mapping failed: expected {len(sequence)} tokens.'
            )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            hidden = esm_model(**encoded, return_dict=True).last_hidden_state[0]
            lysine_token_indices = [
                int(residue_token_indices[position - 1])
                for position in lysine_positions
            ]
            features = hidden[lysine_token_indices].float()
            logits = classifier(features).squeeze(-1)
            ## moves results to cpu from gpu
            probabilities = torch.sigmoid(logits).detach().cpu().tolist()

        records = [
            {
                'predictor': self.name,
                'lysine_position': position,
                'site': f'K{position}',
                'sequence_context': _sequence_context(sequence, position),
                'probability': float(probability),
                'threshold': self.threshold,
                'is_positive': bool(probability > self.threshold),
            }
            for position, probability in zip(lysine_positions, probabilities)
        ]
        result = pd.DataFrame(records)


        result = result.sort_values('probability', ascending=False).reset_index(drop=True)

        paths = ProjectPaths.from_root(Path.cwd())
        result.to_csv(paths.results / "ubi_predictions.csv", index=False)