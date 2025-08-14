# SWE-Lancer

This repo contains the dataset and code for the paper "SWE-Lancer: Can Frontier
LLMs Earn \$1 Million from Real-World Freelance Software Engineering?".

## Setup

### Package Management and Requirements

Install SWELancer with [uv](https://docs.astral.sh/uv/)

```bash
UV_GIT_LFS=1 uv sync
```

### Docker Setup

Tasks in SWELancer rely on Docker images. You can either run SWELancer with a
single "monolith" image, where task-specific setup will occur at run-time, or
you can build task-specific images for each task.

To build task-specific images (e.g., for issue `28565_1001`):

```bash
uv run python scripts/build_images.py 28565_1001 --skip-push
```

Or omit the issue entirely to build images for all tasks:

```bash
uv run python scripts/build_images.py --skip-push
```

Note that while there is some concurrency here, you may want to write your own
script for bulk-building and pushing per-task images, depending on your systems.
Each task image takes 10-20 minutes to build, and occupies ~14GB

To build the "monolith" image run

```bash
uv run python scripts/build_images.py monolith --skip-push
```

To use it, make sure to set `swelancer.use_single_image=True` in your run
command, below.

Note, if you do not want to skip pushing the images, then you should remove the
`--skip-push` argument and provide a `--registry` argument.

Note, we have also already pushed the images to dockerhub, which can be found
[here](https://hub.docker.com/orgs/swelancer/repositories).

Note, to run SWE Manager tasks, you must use the "monolith" image. 

### Environment variables

There are a couple of environment variables SWELancer relies on.

To set them, locate the `sample.env` file in the root directory. This file
contains template environment variables needed for the application:

```plaintext
# sample.env contents example:
USE_WEB_PROXY=false
# ... other variables
```

Create a new file named `.env` and copy the contents from `sample.env`, setting
the appropriate values for each variable. This file will be used to configure
the application. Usually you just need to set `OPENAI_API_KEY` or
`OPENROUTER_API_KEY` to use the OpenAI or OpenRouter API, respectively.

## Running SWELancer

> Note: SWELancer is designed to run with internet disabled. At the moment,
> this is only supported on Linux systems, as it relies on `iptables`. To run
> SWELancer on MacOS, you will need to enable internet access. Simply add the
> `swelancer.disable_internet=False` argument to any of the run commands below.
> Note that running swelancer with internet enabled is not well tested.

We've implemented a [DummySolver](swelancer/solvers/dummy/solver.py) which can
be used to verify that the evaluation works as intended:

- Enabling the `test_user_tool` argument will make the dummy solver use and log
  the user tool. This can take a while to execute (5-10 mins)
- Enabling the `apply_gold_solution` argument will make the dummy solver apply
  the "gold" solution to the task. By default the dummy solver doesn't modify
  the codebase, so you should expect it to fail unless you've enabled this
  argument

Below is an example of how to run the dummy solver on IC SWE Diamond, when not
having it test the user tool, but making it apply the gold solution:

```bash
uv run python swelancer/run_swelancer.py \
  swelancer.split=diamond \
  swelancer.task_type=ic_swe \
  swelancer.solver=swelancer.solvers.dummy.solver:DummySolver \
  swelancer.solver.test_user_tool=False \
  swelancer.solver.apply_gold_solution=True \
  swelancer.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
  swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  swelancer.solver.computer_runtime.env.pull_from_registry=True \
  swelancer.docker_image_prefix=swelancer/swelancer_x86 \
  swelancer.docker_image_tag=releasev1 \
  runner.concurrency=20 \
  runner.experimental_use_multiprocessing=False \
  runner.enable_slackbot=False \
  runner.recorder=nanoeval.recorder:dummy_recorder \
  runner.max_retries=3
```

Same as above but for a single issue (e.g., `28565_1001`):

```bash
uv run python swelancer/run_swelancer.py \
  swelancer.split=diamond \
  swelancer.task_type=ic_swe \
  swelancer.taskset="['28565_1001']" \
  swelancer.solver=swelancer.solvers.dummy.solver:DummySolver \
  swelancer.solver.test_user_tool=False \
  swelancer.solver.apply_gold_solution=True \
  swelancer.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
  swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  swelancer.solver.computer_runtime.env.pull_from_registry=True \
  swelancer.docker_image_prefix=swelancer/swelancer_x86 \
  swelancer.docker_image_tag=releasev1 \
  runner.concurrency=20 \
  runner.experimental_use_multiprocessing=False \
  runner.enable_slackbot=False \
  runner.recorder=nanoeval.recorder:dummy_recorder \
  runner.max_retries=3
```

We've implemented a
[SimpleAgentSolver](swelancer/solvers/swelancer_agent/solver.py) which interacts
with the codebase over multiple steps to solve the task:

- Change the `model` argument to use different models. OAI and OpenRouter models
  are supported. The `model` argument should follow the format
  `<PROVIDER>/<MODEL>`, for example, `openai/gpt-4o`, or
  `openrouter/anthropic/claude-3.5-sonnet`

Below is an example of how to run the `SimpleAgentSolver` on IC SWE Diamond with
gpt-4o:

```bash
uv run python swelancer/run_swelancer.py \
  swelancer.split=diamond \
  swelancer.task_type=ic_swe \
  swelancer.taskset="['28565_1001']" \
  swelancer.solver=swelancer.solvers.swelancer_agent.solver:SimpleAgentSolver \
  swelancer.solver.model=openai/gpt-4o \
  swelancer.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
  swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  swelancer.solver.computer_runtime.env.pull_from_registry=True \
  swelancer.docker_image_prefix=swelancer/swelancer_x86 \
  swelancer.docker_image_tag=releasev1 \
  runner.concurrency=4 \
  runner.experimental_use_multiprocessing=False \
  runner.enable_slackbot=False \
  runner.recorder=nanoeval.recorder:dummy_recorder \
  runner.max_retries=3
```

To run manager tasks, set `swelancer.task_type=swe_manager`. Currently, 
swe_manager tasks are only supported using a monolith image, so you must also 
set `swelancer.use_single_image=True`. Below is an example of how to run 
`SimpleAgentSolver` on SWE Manager Diamond with gpt-4o for a specific task 
(e.g. `16921-manager-0`):

```bash
uv run python swelancer/run_swelancer.py \
  swelancer.split=diamond \
  swelancer.task_type=swe_manager \
  swelancer.taskset="['16921-manager-0']" \
  swelancer.solver=swelancer.solvers.swelancer_agent.solver:SimpleAgentSolver \
  swelancer.solver.model=openai/gpt-4o \
  swelancer.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
  swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  swelancer.solver.computer_runtime.env.pull_from_registry=True \
  swelancer.docker_image_prefix=swelancer/swelancer_x86 \
  swelancer.docker_image_tag=releasev1 \
  swelancer.use_single_image=True \
  runner.concurrency=4 \
  runner.experimental_use_multiprocessing=False \
  runner.enable_slackbot=False \
  runner.recorder=nanoeval.recorder:dummy_recorder \
  runner.max_retries=3
```

### Logging

By default logs appear under the `runs` directory. Each evaluation is identified
as a "run group" and has a unique run group ID; a directory for each run group
is made under the `runs` directory, and a directory for each evaluated task is
made inside the run group directory.
