from abc import ABC
from typing import Self

from nanoeval.solvers.computer_tasks.code_execution_interface import NetworkMode
from nanoeval.solvers.computer_tasks.task import ComputerTask
from pydantic import Field, model_validator


class CompatibleComputerTask(ComputerTask, ABC):
    """
    This class exposes a restricted subset of ComputerTask features that will likely work the same between Alcatraz
    and similar systems.

    This class isn't the be-all-end-all for what features are allowed, but if you can avoid using the features listed
    below, please do.
    """

    enable_cyber_only_features: bool = Field(
        description="ONLY enable if on cyber, which has special additional validations for some features.",
        default=False,
    )

    @model_validator(mode="after")
    def _check_for_compatibility(self) -> Self:
        # Things that definitely don't work the same way
        # Some context on why enabling Internet access is challenging:
        #
        # The only class of eval which is truly "portable" is one that doesn't require external Internet access at all. This is definitely a large restriction in the design space. My guess is that this matters much more for setup scripts than actually permitting the model network access (e.g., maybe we could disable internet during setup, but re-enable for agent rollout), but I think it seems simpler to drop network access fully.
        assert self.network_mode == NetworkMode.NONE, (
            "Internet must be disabled for compatibility reasons"
        )
        assert self.volumes_config == {} and not self.volume_mounts, "Volume config not supported"
        assert not self.side_images

        if not self.enable_cyber_only_features:
            assert not self.docker_compose_yaml

        # Not supported; might work but untested so let's deny until someone needs it.
        assert (
            not self.mem_limit
            and not self.shm_size
            and not self.timeout
            and not self.alcatraz_limits
            and not self.jupyter_setup
            and not self.disk_mount_path
        )

        # Requirements for CR sharing
        assert (
            self.docker_image
            and self.docker_image.startswith("docker.io")
            or self.docker_image == "jupyter/base-notebook:3.11"
        ), "Image must be specified and use a shared CR"

        return self
