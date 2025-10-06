"""Core functionality."""

from .hooks import HookManager, get_hooks
from .registry import ModelRegistry, get_registry

__all__ = ["ModelRegistry", "get_registry", "HookManager", "get_hooks"]
