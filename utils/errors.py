class Error(Exception):
    """Base class for other exceptions."""
    pass
    
class FileFormatNotSupportedError(Error):
    """Raised when attempting to load a file format that is not supported."""
    pass