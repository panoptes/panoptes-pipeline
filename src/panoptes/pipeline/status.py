"""The stages a frame and an observation move through.

These describe *this* pipeline's work -- `CALIBRATING`, `SOLVING`, `MATCHING`,
`EXTRACTING` -- and they used to live in `panoptes-data`, the client that reads
what this pipeline produces. That is the wrong way round: the reader held the
writer's vocabulary, so the writer could not be changed without changing the
reader, and the writer did not import them at all once the Firestore path was
deleted. They belong here, versioned with the data contract. See data contract
5.2.

Both are `IntEnum` because the comparison is the point: the idempotency rule is
"at or past `PROCESSING`", not a set membership test. The order below is
therefore load-bearing, and the members before `PROCESSING` are the ones that
mean "not done" -- including `ERROR`, so a failed frame compares as
unprocessed and gets picked up again.
"""

from __future__ import annotations

from enum import IntEnum, auto


class ImageStatus(IntEnum):
    """The stage one frame has reached.

    Anything below `PROCESSING` means the frame still needs work.
    """

    ERROR = auto()
    MASKED = auto()
    UNKNOWN = auto()
    RECEIVING = auto()
    RECEIVED = auto()
    UNSOLVED = auto()
    PROCESSING = auto()
    CALIBRATING = auto()
    CALIBRATED = auto()
    SOLVING = auto()
    SOLVED = auto()
    MATCHING = auto()
    MATCHED = auto()
    EXTRACTING = auto()
    EXTRACTED = auto()


class ObservationStatus(IntEnum):
    """The stage one observation has reached."""

    ERROR = auto()
    NOT_ENOUGH_FRAMES = auto()
    UNKNOWN = auto()
    CREATED = auto()
    RECEIVING = auto()
    RECEIVED = auto()
    PROCESSING = auto()
    CALIBRATING = auto()
    CALIBRATED = auto()
    MATCHING = auto()
    MATCHED = auto()


def image_status(name: str | None) -> ImageStatus:
    """Read a status name from a document, tolerating what is not there.

    A document written by an older pipeline, or one that never recorded a
    status, reads as `ImageStatus.UNKNOWN` rather than raising -- `UNKNOWN`
    sorts below `PROCESSING`, so an unreadable status means the frame is
    reprocessed. Failing closed here would strand every frame written before
    statuses were recorded.
    """
    if name is None:
        return ImageStatus.UNKNOWN
    try:
        return ImageStatus[name]
    except KeyError:
        return ImageStatus.UNKNOWN
