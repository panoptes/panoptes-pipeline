from enum import IntEnum, auto


class SequenceStatus(IntEnum):
    RECEIVING = 0
    RECEIVED = 10


class ImageStatus(IntEnum):
    RECEIVING = 0
    RECEIVED = 10
    CALIBRATING = 20
    CALIBRATED = 30
    SOLVING = 40
    SOLVED = 50
    MATCHING = 60
    MATCHED = 70
