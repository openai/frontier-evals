# Preparedness Evals

This repository contains the code for multiple Preparedness evals that use nanoeval and alcatraz.

## System requirements

1. Python 3.11 (3.12 is untested; 3.13 will break [chz](https://github.com/openai/chz))

## Install pre-requisites

```bash
for proj in nanoeval alcatraz nanoeval_alcatraz; do
    pip install -e project/"$proj"
done
```

## Evals

- [PaperBench](./project/paperbench/README.md): based on [https://openai.com/index/paperbench/](https://openai.com/index/paperbench/)
- SWELancer (Forthcoming): based on [https://openai.com/index/swe-lancer/](https://openai.com/index/swe-lancer/)
- MLE-bench (Forthcoming): based on [https://openai.com/index/mle-bench/](https://openai.com/index/mle-bench/)
- SWE-bench (Forthcoming): based on [https://openai.com/index/introducing-swe-bench-verified/](https://openai.com/index/introducing-swe-bench-verified/)
