from abc import ABC, abstractmethod

import chz


@chz.chz
class CompatibilityAPI(ABC):
    """
    These functions should be used when writing new evals. These functions represent divergences between supported eval environments
    """

    @property
    @abstractmethod
    def blobstore_root_write(self) -> str:
        """
        Use as the root folder for write access to blob storage via BlobFile.

        You can assume that data written to this path is readable in the current process, but it MAY not be readable
        at the same path after the process exits. So for example, you should not use this function to store datasets
        (use `blobstore_root_read` for that instead).
        """
        pass

    @property
    @abstractmethod
    def blobstore_root_read(self) -> str:
        """
        Use for read-only access to blob storage.

        You should assume data at this path is NOT writable.
        """
        pass
