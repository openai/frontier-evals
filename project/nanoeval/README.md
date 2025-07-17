# nanoeval

Simple, ergonomic, and high performance evals.

# Confidentiality note

`nanoeval` and its dependencies are shared with select Tier II contractors. As such, internal dependencies should be
kept to a minimum as new dependencies require review; this is enforced via tests.

# Principles

1. **Minimal indirection.** You should be able to implement and understand an eval in 100 lines.
2. **Separation of concerns.** Keep data loading away from completions/parsing/different ways of running an eval.
3. **Fast iteration and testability.** nanoevals should import in less than a second and be testable without a live LLM backend.
4. **High performance**. Nanoeval should max out the compute resources available to it.

# Primitives

- `Eval` - A chz class. Enumerates a set of tasks, and (typically) uses a "Solver" to solve them and then records the
  results. Can be configured in code or on the CLI using a chz entrypoint.
- `EvalSpec` - An eval to run and runtime characteristics of how to run it (i.e. concurrency, recording, other
  administrivia)
- `Task` - A separable, scoreable unit of work.
- `Solver` - A strategy (usually involving sampling a model) to go from a task to a result that can be scored. For
  example, there may be different ways to prompt a model to answer a multiple-choice question (i.e. looking at logits,
  using consensus, etc)

# Running your first eval

See [gpqa_api.py](nanoeval/examples/gpqa_api.py) for an implementation of GPQA using the OpenAI API in less than 70 lines of code.

# Features

## Core execution flow

At the highest level: nanoeval is just a library. You can import it and call `nanoeval.run()` on an EvalSpec. nanoeval
then loads all the tasks and runs `eval.evaluate()` in parallel using asyncio.

More granularly: nanoeval operates like a tiny distributed system. Eval state is tracked in a per-process sqlite
database in `/dev/shm` (or `/tmp` on macOS). When you call `.run()`, it queues up the eval and all of its tasks in
sqlite. It then starts one or more executors that continually poll the db for new tasks, run them, and put the results
back in the db.

The executors can operate in two modes:

1. **In-process:** The executor is just an async task running in the same process as the main eval script. The default.
2. **Multiprocessing:** Starts a pool of executor processes that all poll the db. Use this via
   `spec.runner.experimental_use_multiprocessing=True`.

## How to accomplish common usage patterns ("eval a checkpoint", "eval a series of checkpoints")

1. Do it yourself by defining a script with `nanoeval_entrypoint()` that calls `nanoeval.run()`.
   See [nanoeval/examples/gpqa_simple.py](nanoeval/examples/gpqa_simple.py) for an example of this.

# **Writing your first eval**

An eval is just a `chz` class that defines `get_name()`, `get_tasks()`, `evaluate()` and `get_summary()`. Start with
`gpqa_simple.py`; copy it and modify it to suit your needs. If necessary, drop down to the base `nanoeval.Eval` class
instead of using `MCQEval`.

The following sections describe common use case needs and how to achieve them.

## **Public API**

You may import code from any `nanoeval.*` package that does not start with an underscore. Functions and classes that
start with an underscore are considered private; if I refactor them, I will not make changes to your code (I'll # type:
ignore it and ping you).

## Evallib compatibility

Nanoeval uses the same recorder as evallib, so you can use all the same tools (evalboard, etc.) for visualization. I am
planning to make an evallib compat wrapper, so you can include nanoevals in evallib eval sets, but I haven't done that
yet.

# Debugging

Is your big eval not working? Check here. Lots of these are general and will apply to non-nanoeval workloads as well.

## Killing old executors

Sometimes, if you ctrl-c the main job, executors don’t have time to exit. A quick fix:

```bash
pkill -f multiprocessing.spawn
```

## Observability

### py-spy/aiomonitor

`py-spy` is an excellent tool to figure out where processes are stuck if progress isn’t happening. You can check the
monitor to find the PIDs of all the executors and py-spy them one by one. The executors also run `aiomonitor`, so you
can connect to them via `python3 -m aiomonitor.cli ...` to inspect async tasks.
