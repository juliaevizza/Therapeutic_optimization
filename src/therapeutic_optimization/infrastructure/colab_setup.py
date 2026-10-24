"""Review draft: quiet, one-call setup for a Google Colab GPU runtime.

Start with a fresh GPU runtime and run setup before importing workflow or model
libraries. Once this file is on Colab's local disk::

    import colab_setup as file
    setup = file.run()

In IPython, run() publishes notebook imports only after setup succeeds. Outside
IPython, use setup.imports. Importing this file itself performs no installation.
See colab_set_README.md beside this file for the Drive-first notebook cell.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from importlib import import_module, invalidate_caches
from pathlib import Path
from types import SimpleNamespace
from typing import MutableMapping
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback


@dataclass(frozen=True)
class SetupConfig:
    """Maintainer update surface; defaults are a candidate, not a tested lockfile.
    Change immutable source revisions deliberately. HF revisions resolve once
    per profile/runtime and are subsequently loaded from local snapshot paths.
    Supply constraints files to lock transitive dependencies after GPU testing.
    """
    project_url: str = "https://github.com/juliaevizza/Therapeutic_optimization.git"
    # TODO: Update this commit when newer project changes on GitHub should be used.
    # Used only without repo_dir; the notebook passes its current local clone.
    project_revision: str = "efdf449ad41d8abd60bb256b881cd21a400cecca"
    eup_url: str = "https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction.git"
    eup_revision: str = "422f28189c3b3a26d79512a71a160a783daff1d8"
    eup_checkpoint: str = "Model/DNNLinerModel/DNNLinermodel_checkpoint_epoch_34.pth"
    eup_model: str = "facebook/esm2_t36_3B_UR50D"
    eup_model_revision: str = "476b639933c8baad5ad09a60ac1a87f987b656fc"
    esm2_model: str = "facebook/esm2_t33_650M_UR50D"
    esm2_model_revision: str = "08e4846e537177426273712802403f7ba8261b6c"
    # The current adapters use the Transformers 4 API (including torch_dtype).
    notebook_requirements: tuple[str, ...] = ("torch>=2.6", "transformers>=4.57,<5")
    fold_requirements: tuple[str, ...] = (
        "colabfold[alphafold,openmm]==1.6.3", "jax[cuda12]", "openmm[cuda12]",
    )
    # Optional paths to pip constraints .txt files, not inline package strings.
    # Each file lists allowed dependency versions; None adds no constraints file.
    # Keep separate files because notebook and ColabFold use separate environments.
    notebook_constraints: str | None = None
    fold_constraints: str | None = None
    fold_model_type: str = "alphafold2_ptm"
    root: Path = Path("/content/therapeutic_setup")
    timeout_seconds: int = 7200
    minimum_free_gb: float = 25.0


@dataclass(repr=False)
class SetupResult:
    """Paths, imports and configured workflow defaults for the notebook."""
    repo_dir: Path
    eup_dir: Path
    model_cache: Path
    colabfold_executable: Path
    log_path: Path
    manifest_path: Path
    workflow_config: object
    imports: SimpleNamespace

    def __repr__(self) -> str:
        return "SetupResult(ready=True)"


class SetupError(RuntimeError):
    """A concise user-facing failure; details remain in setup.log."""


def _local_path(path: Path) -> Path:
    """Validate setup paths against the standard Colab /content/drive mount.

    This guard does not validate paths supplied separately to the workflow.
    The notebook sets project_root="/content/therapeutic_run" for local queries
    and computation. Drive is a separate destination for selected final results.
    """
    resolved = path.expanduser().resolve()
    content = Path("/content")
    drive = content / "drive"
    if content not in resolved.parents or resolved == drive or drive in resolved.parents:
        raise SetupError("Use a local folder under /content, outside /content/drive.")
    return resolved


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _command(args, log, *, cwd=None, env=None, timeout=7200) -> str:
    """No shell interpolation and no pip/git/download chatter in the notebook."""
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n$ {list(map(str, args))!r}\n")
        stream.flush()
        # File-backed output keeps subprocess chatter out of the notebook.
        offset = stream.tell()
        try:
            result = subprocess.run(
                list(map(str, args)), cwd=cwd, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SetupError(f"Could not finish {args[0]}. See {log}.") from exc
    if result.returncode:
        raise SetupError(f"Setup command failed (exit {result.returncode}). See {log}.")
    with log.open(encoding="utf-8", errors="replace") as stream:
        stream.seek(offset)
        return stream.read()


def _python(python, code, log, *args, timeout=7200) -> str:
    return _command([python, "-c", code, *args], log, timeout=timeout)


def check_runtime(config: SetupConfig, log: Path) -> None:
    if platform.system() != "Linux" or sys.version_info < (3, 10):
        raise SetupError("This draft requires a Linux Colab runtime with Python 3.10+.")
    if not shutil.which("nvidia-smi"):
        raise SetupError("Select Runtime > Change runtime type > GPU, then rerun setup.")
    _command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], log)
    # A rerun already has model files on disk; do not demand their space twice.
    required_gb = 2.0 if (config.root / "model_snapshots.json").is_file() else config.minimum_free_gb
    if shutil.disk_usage(config.root).free < required_gb * 1024**3:
        raise SetupError(f"Setup needs at least {required_gb:g} GB of free local disk.")


def configure_local_storage(root: Path) -> dict[str, Path]:
    paths = {name: _local_path(root / name) for name in ("cache", "huggingface", "torch", "pip")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "HF_HOME": str(paths["huggingface"]),
        "HF_HUB_CACHE": str(paths["huggingface"] / "hub"),
        "TRANSFORMERS_CACHE": str(paths["huggingface"] / "hub"),
        "TORCH_HOME": str(paths["torch"]),
        "XDG_CACHE_HOME": str(paths["cache"]),
        "PIP_CACHE_DIR": str(paths["pip"]),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    })
    return paths


def ensure_system_tools(log: Path) -> None:
    missing = [name for name in ("git", "git-lfs") if shutil.which(name) is None]
    if missing:
        _command(["apt-get", "-qq", "update"], log)
        _command(["apt-get", "-qq", "install", "-y", *missing], log)


def prepare_repository(url: str, revision: str, destination: Path, log: Path) -> str:
    """Fetch an exact Git commit into a managed directory; never reset user edits."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise SetupError("Repository revisions must be full immutable Git commit hashes.")
    environment = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    if not destination.exists():
        destination.mkdir(parents=True)
        _command(["git", "init", destination], log)
        _command(["git", "-C", destination, "remote", "add", "origin", url], log)
    if not (destination / ".git").is_dir():
        raise SetupError(f"Existing folder is not a managed Git checkout: {destination}")
    origin = _command(["git", "-C", destination, "remote", "get-url", "origin"], log).strip()
    if origin != url:
        raise SetupError(f"Repository URL differs at {destination}; choose a new setup root.")
    dirty = _command(["git", "-C", destination, "status", "--porcelain"], log).strip()
    if dirty:
        raise SetupError(f"Preserve your changes in {destination}; setup will not overwrite them.")
    # An interrupted first fetch has no HEAD yet; safely retry that fetch.
    head_file = destination / ".git" / "HEAD"
    detached = head_file.read_text().strip() if head_file.exists() else ""
    if detached != revision:
        _command(["git", "-C", destination, "fetch", "--depth=1", "origin", revision],
                 log, env=environment)
        _command(["git", "-C", destination, "checkout", "--detach", "FETCH_HEAD"],
                 log, env=environment)
    actual = _command(["git", "-C", destination, "rev-parse", "HEAD"], log).strip()
    if actual != revision:
        raise SetupError(f"Unexpected repository revision at {destination}.")
    return actual


def install_notebook_dependencies(repo: Path, config: SetupConfig, log: Path) -> None:
    """Install into a fresh notebook session before importing model libraries."""
    if not (repo / "pyproject.toml").is_file():
        raise SetupError(f"Missing project dependency metadata: {repo}")
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
               "-e", f"{repo}[eup,esm2]", *config.notebook_requirements]
    if config.notebook_constraints:
        command += ["-c", config.notebook_constraints]
    _command(command, log, timeout=config.timeout_seconds)
    invalidate_caches()


def install_colabfold(config: SetupConfig, log: Path) -> Path:
    """Separate JAX/ColabFold dependencies from notebook PyTorch dependencies."""
    environment = _local_path(config.root / "colabfold_env")
    python = environment / "bin" / "python"
    if not python.is_file():
        # Colab images may lack ensurepip. Bootstrap with the host pip instead.
        _command([sys.executable, "-m", "venv", "--without-pip", environment], log)
    _command([sys.executable, "-m", "pip", "--python", python, "install", "pip"], log)
    command = [python, "-m", "pip", "install", "--disable-pip-version-check", *config.fold_requirements]
    if config.fold_constraints:
        command += ["-c", config.fold_constraints]
    _command(command, log, timeout=config.timeout_seconds)
    _command([python, "-m", "pip", "check"], log)
    return python


def prepare_eup(config: SetupConfig, log: Path) -> Path:
    repo = _local_path(config.root / "EUP")
    prepare_repository(config.eup_url, config.eup_revision, repo, log)
    relative = Path(config.eup_checkpoint)
    checkpoint = (repo / relative).resolve()
    if relative.is_absolute() or repo not in checkpoint.parents:
        raise SetupError("EUP checkpoint must be a relative path inside its repository.")
    _command(["git", "-C", repo, "lfs", "install", "--local"], log)
    _command(["git", "-C", repo, "lfs", "pull", f"--include={relative.as_posix()}", "--exclude="],
             log, timeout=config.timeout_seconds)
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1000:
        raise SetupError(f"EUP checkpoint download is incomplete: {checkpoint}")
    with checkpoint.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise SetupError(f"EUP checkpoint is still a Git LFS pointer: {checkpoint}")
    return checkpoint


def prepare_esm2(config: SetupConfig, log: Path) -> dict[str, str]:
    """Download weights to disk only; never load the 3B and 650M models together."""
    record = config.root / "model_snapshots.json"
    request = [config.eup_model, config.eup_model_revision, config.esm2_model, config.esm2_model_revision]
    previous = _read_json(record)
    # Lock the first resolved SHA on this runtime, even when the requested ref is main.
    locked = previous.get("snapshots", {}) if previous.get("request") == request else {}
    payload = json.dumps({"models": [(config.eup_model, config.eup_model_revision),
                                     (config.esm2_model, config.esm2_model_revision)],
                          "locked": locked, "cache": os.environ["HF_HUB_CACHE"],
                          "record": str(record), "request": request})
    _python(sys.executable, '''
import json, pathlib, sys
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoTokenizer
p = json.loads(sys.argv[1]); snapshots = {}
for model, revision in p["models"]:
    old = p["locked"].get(model)
    if old:
        revision = pathlib.Path(old).name
    info = HfApi().model_info(model, revision=revision)
    safe = any(f.rfilename.endswith(".safetensors") for f in info.siblings)
    # EUP's published 3B model uses two PyTorch .bin shards. Prefer safetensors
    # when supplied upstream; never download TensorFlow or duplicate formats.
    pattern = "*.safetensors" if safe else "pytorch_model*.bin"
    snapshot = snapshot_download(repo_id=model, revision=info.sha, cache_dir=p["cache"],
        allow_patterns=["*.json", "*.txt", pattern])
    if not list(pathlib.Path(snapshot).glob(pattern)):
        raise RuntimeError("No supported PyTorch weights in " + snapshot)
    index = pathlib.Path(snapshot) / ("model.safetensors.index.json" if safe else "pytorch_model.bin.index.json")
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if any(not (pathlib.Path(snapshot) / shard).is_file() for shard in shards):
            raise RuntimeError("Incomplete model shards in " + snapshot)
    AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    snapshots[model] = snapshot
record = pathlib.Path(p["record"]); tmp = record.with_suffix(".tmp")
tmp.write_text(json.dumps({"request": p["request"], "snapshots": snapshots}, indent=2))
tmp.replace(record)
''', log, payload, timeout=config.timeout_seconds)
    return _read_json(record)["snapshots"]


def prepare_colabfold_weights(python: Path, config: SetupConfig, log: Path) -> None:
    # This is the only helper coupled to ColabFold's Python download API.
    _python(python, '''
import sys
from colabfold.download import download_alphafold_params, default_data_dir
download_alphafold_params(sys.argv[1], data_dir=default_data_dir)
''', log, config.fold_model_type, timeout=config.timeout_seconds)


def check_installation(python: Path, checkpoint: Path, snapshots: dict, config: SetupConfig, log: Path) -> None:
    """Separate child processes release each framework's GPU allocation on exit."""
    _python(sys.executable, '''
import sys, torch
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
from therapeutic_optimization.config import WorkflowConfig
from therapeutic_optimization.workflow import ProcessFlowDiagram
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the GPU")
(torch.ones(1, device="cuda") + 1).cpu()
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
state = state.get("state_dict", state)
state = {k.removeprefix("module."): v for k, v in state.items()}
size = AutoConfig.from_pretrained(sys.argv[2], local_files_only=True).hidden_size
if set(state) != {"fc1.weight", "fc1.bias"} or tuple(state["fc1.weight"].shape) != (1, size) or tuple(state["fc1.bias"].shape) != (1,):
    raise RuntimeError("EUP checkpoint and ESM embedding dimensions are incompatible")
''', log, str(checkpoint), snapshots[config.eup_model])
    _python(python, '''
import jax, jax.numpy as jnp
from colabfold.download import download_alphafold_params
if not any(d.platform == "gpu" for d in jax.devices()):
    raise RuntimeError("ColabFold/JAX cannot access the GPU")
(jnp.ones(1) + 1).block_until_ready()
''', log)
    help_text = _command([python.parent / "colabfold_batch", "--help"], log)
    if not all(flag in help_text for flag in ("--num-models", "--num-seeds")):
        raise SetupError("ColabFold CLI changed; review the tSA_colab adapter before continuing.")


def load_notebook_imports(repo: Path, config: SetupConfig, snapshots: dict) -> dict:
    """One compatibility boundary for the current project's module names."""
    invalidate_caches()
    module = import_module("therapeutic_optimization.predictors.tUP1_EUP")
    if not Path(module.__file__).resolve().is_relative_to(repo.resolve()):
        raise SetupError("A different project checkout is already imported. Run setup first in a fresh runtime.")
    # The current EUP adapter does not accept a revision/model path argument.
    # Bind its model constant to the resolved local snapshot, so later inference
    # cannot silently download a different main revision. Prefer a constructor
    # model_path argument when that adapter is next revised.
    if module.ESM_MODEL_NAME not in (config.eup_model, snapshots[config.eup_model]):
        raise SetupError("EUP model identity changed; review the adapter/model pairing.")
    if module.EUP_CHECKPOINT.as_posix() != config.eup_checkpoint:
        raise SetupError("EUP checkpoint path changed; review the setup profile.")
    module.ESM_MODEL_NAME = snapshots[config.eup_model]
    settings = import_module("therapeutic_optimization.config")
    workflow_config = settings.WorkflowConfig(
        ubiquitination=settings.UbiquitinationConfig(
            eup_repo_dir=config.root / "EUP", model_cache_dir=Path(os.environ["HF_HUB_CACHE"])),
        esm2=settings.ESM2AnalysisConfig(
            model_name=snapshots[config.esm2_model], model_cache_dir=Path(os.environ["HF_HUB_CACHE"])),
    )
    return {
        "Path": Path, "np": import_module("numpy"), "pd": import_module("pandas"),
        "torch": import_module("torch"), "display": import_module("IPython.display").display,
        "WorkflowConfig": settings.WorkflowConfig,
        "UbiquitinationConfig": settings.UbiquitinationConfig,
        "ESM2AnalysisConfig": settings.ESM2AnalysisConfig,
        "ProcessFlowDiagram": import_module("therapeutic_optimization.workflow").ProcessFlowDiagram,
        "EUPPredictor": module.EUPPredictor,
        "ColabFoldPredictor": import_module("therapeutic_optimization.predictors.tSA_colab").ColabFoldPredictor,
        "REPO_DIR": repo, "EUP_DIR": config.root / "EUP", "workflow_config": workflow_config,
    }


def run(config: SetupConfig | None = None, *, repo_dir: str | Path | None = None,
        namespace: MutableMapping | None = None,
        quiet: bool = True) -> SetupResult:
    """Prepare Colab; export imports only on success. No inference is run.

    quiet=True keeps normal output to a single readiness line; all diagnostics
    are in setup.log. Pass namespace={} to opt out of IPython global exports.
    Run setup first in a fresh GPU runtime, before importing workflow or model
    libraries. Installation finishes before those imports. There is no automatic
    restart detection; reuse the completed setup for additional sequences.

    repo_dir selects the notebook's existing local Git clone. Its current commit
    is recorded and used as-is; project_revision applies only when repo_dir is
    omitted. EUP and model revisions remain pinned in either case.
    """
    config = config or SetupConfig()
    root = _local_path(Path(config.root))
    # Normalize root once so every helper uses the same resolved local location.
    from dataclasses import replace
    config = replace(config, root=root)
    root.mkdir(parents=True, exist_ok=True)
    log = _local_path(root / "setup.log")
    manifest = _local_path(root / "setup_manifest.json")
    lock = root / ".setup.lock"
    import fcntl  # Linux-only, intentionally deferred until run().
    with lock.open("w") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SetupError("Setup is already running in this runtime.") from None
        _atomic_json(manifest, {"status": "preparing", "started": time.time()})
        try:
            check_runtime(config, log)
            configure_local_storage(root)
            ensure_system_tools(log)
            if repo_dir is None:
                repo = _local_path(root / "Therapeutic_optimization")
                project_revision = prepare_repository(config.project_url, config.project_revision, repo, log)
            else:
                repo = _local_path(Path(repo_dir))
                if not (repo / ".git").is_dir() or not (repo / "pyproject.toml").is_file():
                    raise SetupError(f"Expected a complete local project clone at {repo}.")
                origin = _command(["git", "-C", repo, "remote", "get-url", "origin"], log).strip()
                if origin != config.project_url:
                    raise SetupError(f"Unexpected project repository at {repo}.")
                if _command(["git", "-C", repo, "status", "--porcelain"], log).strip():
                    raise SetupError("The project clone contains local edits; commit or preserve them before setup.")
                project_revision = _command(["git", "-C", repo, "rev-parse", "HEAD"], log).strip()
            install_notebook_dependencies(repo, config, log)
            # Editable installs can add .pth files that this live kernel has not read.
            source = str(repo / "src")
            if source not in sys.path:
                sys.path.insert(0, source)
            fold_python = install_colabfold(config, log)
            checkpoint = prepare_eup(config, log)
            snapshots = prepare_esm2(config, log)
            check_installation(fold_python, checkpoint, snapshots, config, log)
            prepare_colabfold_weights(fold_python, config, log)
            with log.open("a") as stream, redirect_stdout(stream), redirect_stderr(stream):
                imports = load_notebook_imports(repo, config, snapshots)
            # Existing ColabFoldPredictor resolves the CLI using PATH.
            bin_dir = str(fold_python.parent)
            os.environ["PATH"] = os.pathsep.join([bin_dir, *[p for p in os.environ["PATH"].split(os.pathsep) if p != bin_dir]])
            for name, python in (("notebook", sys.executable), ("colabfold", fold_python)):
                freeze = _command([python, "-m", "pip", "freeze"], log)
                (root / f"{name}-resolved.txt").write_text(freeze)
            _atomic_json(manifest, {
                "status": "ready", "project_revision": project_revision,
                "project_directory": str(repo),
                "eup_revision": config.eup_revision,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "model_snapshots": snapshots, "python": sys.version,
                "fold_requirements": config.fold_requirements,
                "note": "Setup checks only; no end-to-end protein prediction has been run.",
            })
        except Exception as exc:
            with log.open("a") as stream:
                traceback.print_exc(file=stream)
            _atomic_json(manifest, {"status": "failed",
                                    "error": str(exc), "log": str(log)})
            if isinstance(exc, SetupError):
                raise
            raise SetupError(f"Setup failed: {exc}. See {log}.") from None
    if namespace is None:
        try:
            from IPython import get_ipython
            shell = get_ipython()
            namespace = shell.user_ns if shell is not None else None
        except ImportError:
            pass
    if namespace is not None:
        namespace.update(imports)
    result = SetupResult(repo, root / "EUP", Path(os.environ["HF_HUB_CACHE"]),
                         fold_python.parent / "colabfold_batch", log, manifest,
                         imports["workflow_config"], SimpleNamespace(**imports))
    print("Setup ready." if quiet else f"Setup ready. Files: {root}; log: {log}")
    return result
