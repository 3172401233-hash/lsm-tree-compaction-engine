class LSMError(Exception):
    """Base error for the LSM engine."""


class CompactionError(LSMError):
    """A compaction task failed or conflicted with in-progress work."""


class RecoveryError(LSMError):
    """The on-disk state could not be recovered."""
