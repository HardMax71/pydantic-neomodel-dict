from typing import Any, Dict, Optional


class ConversionError(Exception):
    """Exception raised for errors during model conversion.

    Attributes:
        message: The error message
        context: Additional context information about the error
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (context: {context_str})"
        return self.message
