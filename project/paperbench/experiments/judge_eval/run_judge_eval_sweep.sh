#!/bin/bash

export OPENAI_API_KEY="<JUDGE_EVAL_API_KEY>"

if [ "$OPENAI_API_KEY" = "<JUDGE_EVAL_API_KEY>" ]; then
  echo "Error: Please set a valid OpenAI API key in the script. Replace <JUDGE_EVAL_API_KEY> with the judge eval API key."
  exit 1
fi

for model in o3-mini-2025-01-31 o1-2024-12-17 o1-mini-2024-09-12; do
  echo "Running judge eval for $model-high"
  python paperbench/scripts/run_judge_eval.py -j simple -m $model \
    --reasoning-effort high \
    --output-dir experiments/judge_eval/judge_eval_results/ \
    --example-ids pinn/0 rice/0 stay-on-topic-with-classifier-free-guidance/0 all-in-one/0 semantic-self-consistency/0
  echo "-----------------------------"
done

for model in gpt-4o-mini-2024-07-18 gpt-4o-2024-08-06; do
  echo "Running judge eval for $model"
  python paperbench/scripts/run_judge_eval.py -j simple -m $model \
    --reasoning-effort none \
    --output-dir experiments/judge_eval/judge_eval_results/ \
    --example-ids pinn/0 rice/0 stay-on-topic-with-classifier-free-guidance/0 all-in-one/0 semantic-self-consistency/0
  echo "-----------------------------"
done

python paperbench/scripts/run_judge_eval.py -j random \
  --output-dir experiments/judge_eval/judge_eval_results/ \
  --example-ids pinn/0 rice/0 stay-on-topic-with-classifier-free-guidance/0 all-in-one/0 semantic-self-consistency/0

python paperbench/scripts/run_judge_eval.py -j dummy \
  --output-dir experiments/judge_eval/judge_eval_results/ \
  --example-ids pinn/0 rice/0 stay-on-topic-with-classifier-free-guidance/0 all-in-one/0 semantic-self-consistency/0

# finally, single run of judge-eval on o3-mini-high with --code-only
# to be able to compare token counts with default PaperBench
python paperbench/scripts/run_judge_eval.py -j simple -m o3-mini-2025-01-31 \
  --reasoning-effort high \
  --output-dir experiments/judge_eval/judge_eval_results/code_only \
  --example-ids pinn/0 rice/0 stay-on-topic-with-classifier-free-guidance/0 all-in-one/0 semantic-self-consistency/0 \
  --code-only
