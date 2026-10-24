# Colab setup

Publish these together on GitHub's `main` branch:

- `src/infrastructure/colab_setup.py`
- `src/infrastructure/colab_set_README.md` (this file)
- `notebooks/therapeutic_optimization_colab.ipynb`

The notebook uses `colab_setup.py` as its single setup entry point. It does not
call the older local `install_dependencies.py` or `scripts/prepare_colab.py`.
The rest of the project, including `pyproject.toml` and workflow modules, must
remain in the repository. Importing the setup module alone installs nothing.

## First executable cell

The supplied notebook saves this cell in Colab's form view (`cellView: form`),
so its code is hidden by default. Users see the title and Run button; the Drive
authorization prompt and status output stay visible. The code below is for
maintainers. If pasting it into a different notebook, select the setup cell,
use **View > Show/hide code** to hide its source, then save the notebook.
The `#@title` line supplies the form title; copying the code alone does not copy
the saved hidden-code metadata. Users can deliberately reveal the source.

Choose a fresh GPU runtime. This cell requests Drive access first, then clones
source to Colab's local disk and starts setup. An already authorized mount may
be reused without another prompt. Do not capture or suppress the mount output.

```python
#@title Connect Drive and prepare the environment
# Authorize Drive before starting the slower downloads and installation.
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/juliaevizza/Therapeutic_optimization.git"
REPO_DIR = Path("/content/Therapeutic_optimization")
LOCAL_RUN_DIR = Path("/content/therapeutic_run")
DRIVE_RESULTS_DIR = Path("/content/drive/MyDrive/Therapeutic_optimization_results")

# Download source code to the local disk, never to Drive.
if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", "--branch", "main",
         REPO_URL, str(REPO_DIR)],
        check=True,
    )

SETUP_DIR = REPO_DIR / "src" / "infrastructure"
if not (SETUP_DIR / "colab_setup.py").is_file():
    raise FileNotFoundError(
        "This clone does not contain src/infrastructure/colab_setup.py. "
        "Publish that file to GitHub main, then use a fresh Colab runtime."
    )
sys.path.insert(0, str(SETUP_DIR))

import colab_setup as file
print("Drive connected. Preparing packages and models; this may take a while.")
setup = file.run(repo_dir=REPO_DIR)
```

The clone uses the current `main` branch when first downloaded. A rerun reuses
the same checkout; it does not pull changes or overwrite edits. After publishing
new project changes, use a fresh runtime to fetch the new branch contents.

`run(repo_dir=REPO_DIR)` installs from that exact local checkout and records its
commit in the setup manifest. It does not download a second project copy or
replace the checkout with the older `SetupConfig.project_revision`. That pin is
only a fallback for standalone calls to `run()` without `repo_dir`; its update
TODO remains beside the setting. EUP and Hugging Face model revisions stay pinned.

## Analysis cell

Setup publishes `ProcessFlowDiagram` and `workflow_config` to the notebook.
Keep that configuration so the predictors use the downloaded local model paths.
Set `UBI_THRESHOLD` and `WT_SEQUENCE` in the intervening notebook cells, then run:

```python
# Keep the local model paths configured by setup.
workflow_config.ubiquitination.threshold = UBI_THRESHOLD

workflow = ProcessFlowDiagram(
    seq=WT_SEQUENCE,
    property="ubiquitination",
    mutant_mode="singular",
    project_root=LOCAL_RUN_DIR,
    alt_AA=("R",),
    config=workflow_config,
    drive_root=DRIVE_RESULTS_DIR,
)

results = workflow.run_all()
results
```

## Storage separation

| Location | Contents |
| --- | --- |
| `/content/Therapeutic_optimization` | Source cloned from GitHub |
| `/content/therapeutic_setup` | EUP checkout, model weights, caches, isolated ColabFold environment, setup logs and manifest |
| `/content/therapeutic_run` | WT and mutant FASTAs, batch query CSVs, MSAs, predictions and intermediate outputs |
| `/content/drive/MyDrive/Therapeutic_optimization_results` | Destination for selected final tables, structures, figures and reports |

Notebook Python packages also live on the Colab VM. Drive authorization does not
change the working directory or package/model cache locations. Build queries and
compute results locally, then copy only the selected user-facing deliverables.
Never export the entire working directory.

The current workflow's `drive_root` creates result folders; it does **not** copy
results into them. Actual selected-results export remains a separate workflow
step. `project_root` must stay at the local path shown above. The setup path guard
covers setup assets; it does not validate arbitrary workflow arguments.

## Installations, imports and reruns

Run setup before importing model or workflow libraries. On a successful run,
installation and weight downloads finish before the notebook imports are exposed.
The intended order is a fresh GPU session, setup, then analysis; there is no
mandatory restart cell and no automatic kernel restart. Colab may preload
libraries, so a real compatibility problem can still require a session restart.
The draft does not automatically detect changed loaded packages.

Installation output is in `/content/therapeutic_setup/setup.log`; the cell shows
brief status messages. Exceptions remain visible. `setup.imports` and
`setup.workflow_config` provide explicit access to the exported objects.

## Maintainer settings

`SetupConfig` centralizes package requirements, EUP/model revisions and caches.
`notebook_constraints` and `fold_constraints` are optional strings containing
paths to pip constraints text files, such as `/content/notebook-constraints.txt`.
Each file lists version rules, one per line (for example `transformers>=4.57,<5`).
The rules restrict versions of packages being installed; they do not request
installation or override incompatible requirements. Both settings default to
`None`. Use separate validated constraint files for the two Python environments.

The EUP adapter currently hardcodes its model name. Setup binds that module
constant to the downloaded local snapshot. A future adapter can accept an
explicit model path instead. A new EUP architecture requires adapter review;
changing an installation version alone cannot make checkpoints compatible.

## Validation boundaries

This is a review draft. Offline tests do not prove that the candidate package
versions work together on Colab's current GPU image. Drive authorization, full
installation, model downloads and actual inference still need a Colab acceptance
run. The separate 650M scoring stage is not called by the reviewed `run_all()`;
its module also has unresolved helper/configuration references. Setup prepares
its dependencies and weights, but does not integrate or repair that stage.

ColabFold may contact its MSA service during prediction. Local query construction
and local output storage do not mean inference is fully offline. The setup
prefetches monomer AlphaFold parameters, not a local MSA database.

References: [Colab Drive authorization](https://research.google.com/colaboratory/faq.html),
[ColabFold installation](https://github.com/sokrypton/ColabFold#installation),
[Hugging Face downloads](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download),
[pip constraints](https://pip.pypa.io/en/stable/user_guide/#constraints-files).
