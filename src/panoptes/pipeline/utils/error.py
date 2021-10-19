class ProcessingError(Exception):
    """Base class for processing errors"""
    pass


class TooFewFrames(ProcessingError):
    pass
