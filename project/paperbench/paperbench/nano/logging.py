import blobfile as bf
import structlog
from nanoeval.library_config import LibraryConfig, set_library_config
from nanoeval.setup import nanoeval_logging
from paperbench.utils import get_default_runs_dir
from structlog.typing import EventDict
from typing_extensions import override


def setup_logging(library_config: LibraryConfig):
    """
    Helper function for setting up logging for nanoeval
    to be called inside the entrypoint function
    """
    set_library_config(library_config)
    nanoeval_logging()


def file_processor(
    logger: structlog.stdlib.BoundLogger,
    log_method: str,
    original_event_dict: EventDict,
) -> EventDict:
    """
    A structlog processor that redirects logs to a file based on the specified
    run_group and run_id
    """

    event_dict = dict(original_event_dict)  # Avoid mutating the original

    destinations = event_dict.pop("destinations", [])
    run_group_id = event_dict.pop("run_group_id", None)
    run_id = event_dict.pop("run_id", None)
    runs_dir = event_dict.pop("runs_dir", get_default_runs_dir())

    if "run" in destinations and run_group_id and run_id:
        dst = bf.join(runs_dir, run_group_id, run_id, "run.log")
        with bf.BlobFile(dst, "a") as f:
            f.write(str(event_dict) + "\n")

    if "group" in destinations and run_group_id:
        dst = bf.join(runs_dir, run_group_id, "group.log")
        with bf.BlobFile(dst, "a") as f:
            f.write(str(event_dict) + "\n")

    return original_event_dict


class PaperBenchLibraryConfig(LibraryConfig):
    """
    To be called at the eval entrypoint with
    nanoeval.library_config.set_library_config(PaperBenchLibraryConfig())
    """

    @override
    def setup_logging(self) -> None:
        super().setup_logging()

        # we are simply overriding the structlog configuration
        # specifically we are adding the `file_processor`
        # for distributing file-specific logs (e.g. logs specific to a certain paper)
        structlog.configure(
            processors=[
                file_processor,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
