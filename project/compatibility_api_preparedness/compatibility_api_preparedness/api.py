import chz
from compatibility_api.api import CompatibilityAPI


@chz.chz
class CompatibilityAPIPreparedness(CompatibilityAPI):
    """
    This is the openai/preparedness implementation of CompatibilityAPI, which abstracts over differences in dev envs.

    Each client repo has its own implementation of this compatibility layer.
    """

    override_use_local_dir: str = chz.field(
        doc="Override the base_dir for this class to use a local dir."
    )

    @chz.init_property
    def _base_dir(self) -> str:
        return self.override_use_local_dir

    @property
    def blobstore_root_write(self) -> str:
        return self._base_dir

    @property
    def blobstore_root_read(self) -> str:
        return self._base_dir
