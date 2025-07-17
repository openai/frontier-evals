## Setup

See the PaperBench [README](../../README.md) for setup instructions.

## Quickstart

See the PaperBench [README](../../README.md) for quickstart instructions.

## Resuming runs

To resume a run group, execute the nanoeval script with the following command line argument:

```bash
paperbench.resume_run_group_id="2024-12-13T01-49-15-UTC_run-group_aisi-basic-agent"  # Replace with your run group id here
```

The `nanoeval` script checks for existing outputs in:

```
{runs_dir}/{run_group_id}/{task_id}/
```

## Logs

The nanoeval script produces three types of logs:
1. **Standard Output & Error**: All outputs from tasks and Alcatraz (very noisy).
2. **Task Logs**: 
   ```
   {runs_dir}/{run_group_id}/{task_id}/eval.log
   ```
   These provide task-specific logs that are easier to read and are helpful to understand how each task is progressing.
3. **Record Logs**: 
   ```
   {nanoeval_dir}/records/{timestamp}.json
   ```
   Structured nanoeval-style records capturing task outputs in a machine-readable format, which is often useful for calculating metrics and plotting results.
