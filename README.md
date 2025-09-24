# Frontier Evals

Code for evals measuring frontier model capabilities.

- [PaperBench](./project/paperbench): End-to-end replication of state-of-the-art AI papers. [Paper](https://arxiv.org/abs/2504.01848) | [Blog](https://openai.com/index/paperbench)
- [SWE-Lancer](./project/swelancer): Real freelance software engineering tasks with end-to-end tests. [Paper](https://arxiv.org/abs/2502.12115) | [Blog](https://openai.com/index/swe-lancer)

### Repository Layout

```
.
├── pyproject.toml             # Shared tooling configuration (Ruff, Black, etc.)
└── project/
    ├── common/               # Shared libraries
    ├── paperbench/           # PaperBench eval
    └── swelancer/            # SWE-Lancer eval
```

Each eval directory has its own `README.md`, `pyproject.toml` and `uv.lock`.

### Requirements

We manage environments with [`uv`](https://github.com/astral-sh/uv). Install `uv` once, then run `uv sync` (or `uv pip install -r ...`) inside the project of interest to create its virtual environment from the checked-in `uv.lock`.

### Running Evals

Each eval directory documents how to reproduce runs, configure models, and interpret results. Start with the suite `README.md`, then consult any scripts under `scripts/` or `runtime_*/` directories for orchestration details. When in doubt:

1. `cd` into the eval directory.
2. `uv sync` to install dependencies.
3. Follow the local instructions in the `README.md`.

### Development Workflow

- Create or activate the environment for the project you are working on with `uv`. Example for PaperBench:
  - `cd project/paperbench`
  - `uv sync`
  - `uv run pytest`
- Code style and linting use Ruff (with autofix profiles in `pyproject.toml` and `project/common/tooling/ruff_autofix_minimal.toml`) and Black. Run `uv run ruff check --fix` or use the provided Poe/`make` tasks where available.
- Shared utilities live under `project/common`; changes there may affect multiple evals. Bump the relevant editable dependencies if you create new shared subpackages.
