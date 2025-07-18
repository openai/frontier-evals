from __future__ import annotations

import structlog

import chz
import nanoeval
from nanoeval.eval import EvalSpec, RunnerArgs
from nanoeval.library_config import LibraryConfig
from nanoeval.recorder import dummy_recorder
from nanoeval.setup import nanoeval_entrypoint
from swelancer.eval import SWELancerEval
from swelancer.utils.custom_logging import (
    setup_logging,
)
from swelancer.utils.custom_logging import (
    swelancer_library_config as default_library_config,
)

default_runner = RunnerArgs(
    concurrency=20,
    experimental_use_multiprocessing=False,
    enable_slackbot=False,
    recorder=dummy_recorder(),
    max_retries=0,
)


async def main(
    swelancer: SWELancerEval,
    runner: RunnerArgs = default_runner,
    library_config: LibraryConfig = default_library_config,
) -> None:
    setup_logging(library_config)
    logger = structlog.stdlib.get_logger(component=__name__)

    report = await nanoeval.run(
        EvalSpec(
            eval=swelancer,
            runner=runner,
        )
    )

    logger.info(f"Eval report: {report}", destinations=["group"], _print=True)


if __name__ == "__main__":
    nanoeval_entrypoint(chz.entrypoint(main))
