# PaperBench

This repo contains the dataset and code for the paper "PaperBench: Evaluating AI's Ability to Replicate AI Research".

PaperBench evaluates AI agents on replicating 20 Spotlight and Oral papers from ICML 2024 from scratch.

Each sample of PaperBench includes a research paper and a rubric that defines the requirements for a successful replication.

PaperBench runs in 3 stages:

1. **Agent Rollout**: The agent is executed in an ubuntu container where it must create its submission: a codebase that replicates the paper.
2. **Reproduction**: The agent's submitted codebase is executed in a fresh second container with GPU access so that results from executing the codebase are obtained. This creates the _executed submission_.
3. **Grading**: The _executed submission_ is graded using paper's rubric. A third container is created where the judge is run.

## Nanoeval and Alcatraz pre-requisites

PaperBench uses [nanoeval](../nanoeval) (a high performance evaluation module) and [alcatraz](../alcatraz) (a containerization module) to orchestrate runs.

First, install nanoeval and alcatraz by following the [instructions](../../README.md) at the root of this monorepo.

## PaperBench Setup

All commands in this README should be run from the [root](./) of the PaperBench project.

### Installation

Install PaperBench by running:

```console
pip install -e .
```

**(Optional)** If you're committing code, you can install the pre-commit hooks by running:

```console
pip install pre-commit
pre-commit install
```

### Get the data

The dataset is stored using [Git-LFS](https://git-lfs.com/). Download and install LFS, then run:

```console
git lfs fetch --all
git lfs pull
```


### Environment variables

PaperBench requires API keys for running the agents and judge. To set up your environment variables, copy the top-level `.env.example` file to a new file named `.env` and fill in the required values.

**Note:** The API key used by the Judge, `GRADER_OPENAI_API_KEY`, defaults to `OPENAI_API_KEY` if it's not set.

Once you have filled in the `.env` file, source it:

```bash
source .env
```

### Agent resources

Some papers require access to the OpenAI API and HuggingFace in order to be fully replicated. To make these API keys available to the agent, first copy `paperbench/agents/agent.env.example` to `paperbench/agents/agent.env` and populate the `OPENAI_API_KEY` and `HF_TOKEN` fields. The `agent.env` is provided in the agent's workspace and the agent is informed of this file in our [default instructions for BasicAgent](paperbench/agents/aisi-basic-agent/templates.py), but `agent.env` is not sourced automatically on behalf of the agent.

The HuggingFace token can be obtained through the website after making an account, and should have authorization to access [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b) and [ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

For convenience, the table below lists which papers require a HuggingFace token or OpenAI API key for it to be fully replicated:

| **Paper ID**                   | **HF Token Needed?**          | **OpenAI API Needed?**                                |
|-------------------------------|--------------------------------|-------------------------------------------------------|
| **longlm**                    | Yes ([Llama-2](https://huggingface.co/meta-llama/Llama-2-7b))                  | No                                                    |
| **lca‑on‑the‑line**           | Yes ([ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k))                 | No                                                    |
| **stochastic‑interpolants**   | Yes ([ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k))                 | No                                                    |
| **test‑time‑model‑adaptation**| Yes ([ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k))              | No                                                    |
| **robust‑clip**               | Yes ([ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k))                 | No                                                    |
| **semantic‑self‑consistency** | No                             | Yes (`gpt‑3.5‑turbo` / `gpt‑4o‑mini`)                 |
| **bbox** | No                             | Yes (`gpt‑3.5‑turbo` / `gpt-4`)                 |

### Build Docker images

Running agents with nanoeval and alcatraz requires Docker, which you can install by following the guide [here](https://docs.docker.com/engine/install/).

**(Optional):** To enable GPU support for the agents, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

PaperBench requires the following Docker images to be built before it can run:

- [paperbench/reproducer.Dockerfile](paperbench/reproducer.Dockerfile): The environment in which the submission is reproduced.
- [paperbench/grader.Dockerfile](paperbench/grader.Dockerfile): The environment in which the judge is run to grade the agent's submission.
- [paperbench/agents/Dockerfile.base](paperbench/agents/Dockerfile.base): The base environment for agents, which contains basic packages like `curl`, `git`, `python`, etc. and a Conda environment that can be used to run the agents.

You will need to build the images for each agent that you want to run. We provide the following agents out-of-the-box:

- [paperbench/agents/dummy/Dockerfile](paperbench/agents/dummy/Dockerfile): A agent for that creates a dummy submission, useful for testing the eval end-to-end.
- [paperbench/agents/aisi-basic-agent/Dockerfile](paperbench/agents/aisi-basic-agent/Dockerfile): Simple ReAct style agents with tools available to them.

For convenience, we’ve provided a [script](paperbench/scripts/build-docker-images.sh) that builds all the above images:

```bash
bash paperbench/scripts/build-docker-images.sh
```

## Set agent configurations

Configurations for each agent are stored in the [paperbench/agents/](paperbench/agents/) directory. Each agent has a `config.yaml` file which sets the agent's possible configurations, e.g. which model to use, how long to run the agent for, etc.

## Quickstart

To see all configurable options for PaperBench, run:

```bash
python -m paperbench.nano.entrypoint --help
```

To use our default solver (`ExternalPythonCodingSolver`), you must set:
- `paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver`, specifying to use the `ExternalPythonCodingSolver` solver.
- `paperbench.solver.agent_id` to the id of the agent you want to run.
- `paperbench.solver.cluster_config` to `alcatraz.clusters.local:LocalConfig`, to use LocalCluster for managing the agent's container.
- `paperbench.solver.cluster_config.image` to the Docker image that corresponds to the agent you want to run.
- `paperbench.solver.is_nvidia_gpu_env=True` if you have a local available NVIDIA gpu and would like the agent to have access to it.
- `runner.recorder` to `nanoeval.json_recorder:json_recorder` to save the results of the nanoeval run to a JSON file.

To test that things are set up correctly, run the dummy agent with the dummy judge (`paperbench.judge.scaffold=dummy`) on the debug split. Note this dummy agent is different to the [default nanoeval "dummy" agent](../nanoeval/nanoeval/solvers/computer_tasks/solver.py); our dummy agent tests several aspects of the infrastructure and makes a toy submission.

```bash
python -m paperbench.nano.entrypoint \
    paperbench.paper_split=debug \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.agent_id=dummy \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image=dummy:latest \
    paperbench.judge.scaffold=dummy \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

To run a development configuration of BasicAgent that uses `gpt-4o-mini` and runs for 5 minutes, use `paperbench.solver.agent_id=aisi-basic-agent-openai-dev`. To run that agent on the full PaperBench dataset, execute:

```bash
python -m paperbench.nano.entrypoint \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.agent_id=aisi-basic-agent-openai-dev \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image=aisi-basic-agent:latest \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

**Note** that we set the default concurrency to 5, you may want to adjust this based on your setup.
To set the concurrency, simply specify `runner.concurrency=<int>` in the command.
**Note** You may use a custom solver: set `paperbench.solver` to point to your custom solver class. This solver should be a `PythonCodingSolver` class.


## Retrieving results

A run group is created for each launch of PaperBench. A run group contains multiple individual runs, one for each paper attempt. Runs are stored in the `runs` directory.

In each run group directory, there is a `group.log` file that contains the logs for that run group.

In each run directory there is:
- `run.log`: The log for that run.
- `status.json`: The status of that run.
- `metadata.json`: Metadata for that run.
- `pb_result.json`: The grading result for that run.

Snapshots from the agent rollout are also stored in the run directory. An initial snapshot is created when the agent starts, and a final snapshot is created when the agent finishes. Intermediate snapshots are created throughout the agent rollout and can be set via `paperbench.solver.upload_interval_messages` or `paperbench.solver.upload_interval_seconds`.

`runs` directory structure:
```
runs/
├── <run_group_id>/
│   ├── <run_id>/
│   │   ├── <inital_snapshot_metadata.json>
│   │   ├── <inital_snapshot.tar.gz>
│   │   ├── ...
│   │   ├── ...
│   │   ├── <final_snapshot_metadata.json>
│   │   ├── <final_snapshot.tar.gz>
│   │   ├── <final_snapshot>_repro.tar.gz>
│   │   ├── <final_snapshot>_repro_grader_output_0.json
│   │   ├── <final_snapshot>_repro_metadata.json
│   │   ├── metadata.json
│   │   ├── pb_result.json
│   │   ├── status.json
│   │   └── run.log
│   ├── group.log
│   └── <other_run_ids>/...
└── <other_run_group_ids>/...
```

## PaperBench Code-Dev

**PaperBench Code-Dev** is a lighter-weight variant of PaperBench. Unlike the full PaperBench pipeline -- which involves executing the agent’s submission in a separate reproduction step -- PaperBench Code-Dev skips the reproduction step and only grades the agent's submission on the **Code Development** requirements. This means:

- The Judge only checks **Code Development** requirements (e.g., “Is there an implementation of method X?”). It skips checking Execution requirements that check that the code runs correctly, and skips checking Result Match requirements that check that the paper’s empirical results have been replicated.
- You **don’t need a GPU to run the reproduction step** where the agent's submission is executed. This often reduces cost and runtime significantly.
- There is **less of a need to make a GPU available to the agent** when it is creating its submission. Although having access to a GPU is helpful for the agent to run intensive experiments that verify that its code is correct, the agent can get away with less end-to-end testing of its code since it is only graded on **Code Development** requirements.

We think PaperBench Code-Dev offers a convenient, lower-cost, but less rigorous way of assessing paper replication. It doesn't require GPUs and typically cuts grading costs (we’ve seen around an 85% reduction in o3-mini SimpleJudge costs for the average submission), making it a accessible alternative for assessing models' abilities to replicate papers.

To run the Code-Dev variant, simply include the following flag:

```bash
paperbench.judge.code_only=True
```

## Dataset

The PaperBench dataset is stored in the [papers](data/papers) directory. Each paper has its own directory, which contains:
- The paper in PDF and Markdown format.
- `addendum.md` containing information from the author that is helpful for replication.
- `assets` directory, containing necessary resources for replication. For example, images from the paper are includes in this directory.
- `rubric.json` file that is used to grade submissions
- `blacklist.txt` containing websites that the agent is disallowed from using (e.g. the paper's original codebase).
- `config.yaml` file that defines the paper's id.
- `judge.addendum.md` containing information from the author that is helpful for the judge to know whilst grading submissions. Not every paper has a judge addendum

### Viewing rubrics

We include a web app to view and edit rubrics. To view a paper's rubric that is in `PAPER_DIR` with filename `RUBRIC_FILE_NAME`:

```bash
python paperbench/gui/app.py --path-to-paper <PAPER_DIR> --rubric-file-name <RUBRIC_FILE_NAME>
```

To view a rubric that has been graded (see below), pass the `--graded` flag:

```bash
python paperbench/gui/app.py --path-to-paper <PAPER_DIR> --rubric-file-name <RUBRIC_FILE_NAME> --graded
```

## JudgeEval

We've created an auxiliary evaluation, [JudgeEval](paperbench/judge/judge_eval), to evaluate the accuracy of judges. See the JudgeEval [README](paperbench/judge/judge_eval/README.md) for more information.

## Authors
Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Chan Jun Shern, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, Tejal Patwardhan

## Citation

Please cite using the following BibTeX entry:

```
@misc{starace2025paperbenchevaluatingaisability,
      title={PaperBench: Evaluating AI's Ability to Replicate AI Research}, 
      author={Giulio Starace and Oliver Jaffe and Dane Sherburn and James Aung and Jun Shern Chan and Leon Maksin and Rachel Dias and Evan Mays and Benjamin Kinsella and Wyatt Thompson and Johannes Heidecke and Amelia Glaese and Tejal Patwardhan},
      year={2025},
      eprint={2504.01848},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.01848}, 
}
```