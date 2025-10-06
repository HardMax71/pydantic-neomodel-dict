import logging
from threading import RLock
from typing import Callable, List, Union

from neomodel import StructuredNode
from neomodel.async_.core import AsyncStructuredNode

logger = logging.getLogger(__name__)


NodeT = Union[StructuredNode, AsyncStructuredNode]


class HookManager:
    """Thread-safe hook management.

    Hooks are callbacks that execute during node lifecycle events.

    Hook Execution Model:
    - before_* hooks: Exceptions abort the operation (validation)
    - after_* hooks: Exceptions are logged but don't abort (notifications)
    - Hooks execute SERIALLY under lock for automatic thread safety
    - This means hooks are safe to modify shared state without extra locking

    Performance Note:
    - Hooks execute one at a time (serialized) to guarantee thread safety
    - Avoid long-running operations in hooks as they block other threads
    - For async operations, use after_* hooks and handle asynchronously
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._before_save: List[Callable[[NodeT], None]] = []
        self._after_save: List[Callable[[NodeT], None]] = []
        self._before_connect: List[Callable[[NodeT, str, NodeT], None]] = []
        self._after_connect: List[Callable[[NodeT, str, NodeT], None]] = []

    def register_before_save(self, hook: Callable[[NodeT], None]) -> None:
        """Register before-save hook.

        Hooks execute serially under lock (thread-safe by default).
        Exceptions abort save operation.
        """
        with self._lock:
            self._before_save.append(hook)

    def register_after_save(self, hook: Callable[[NodeT], None]) -> None:
        """Register after-save hook.

        Hooks execute serially under lock (thread-safe by default).
        Exceptions are logged but don't abort.
        """
        with self._lock:
            self._after_save.append(hook)

    def register_before_connect(self, hook: Callable[[NodeT, str, NodeT], None]) -> None:
        """Register before-connect hook.

        Hooks execute serially under lock (thread-safe by default).
        Exceptions abort connection.
        """
        with self._lock:
            self._before_connect.append(hook)

    def register_after_connect(self, hook: Callable[[NodeT, str, NodeT], None]) -> None:
        """Register after-connect hook.

        Hooks execute serially under lock (thread-safe by default).
        Exceptions are logged but don't abort.
        """
        with self._lock:
            self._after_connect.append(hook)

    def execute_before_save(self, node: NodeT) -> None:
        """Execute before-save hooks. First failure aborts.

        Hooks execute serially under lock for thread safety.
        """
        with self._lock:
            for hook in self._before_save:
                hook(node)  # Let exceptions propagate

    def execute_after_save(self, node: NodeT) -> None:
        """Execute after-save hooks. Failures are logged.

        Hooks execute serially under lock for thread safety.
        """
        with self._lock:
            for hook in self._after_save:
                try:
                    hook(node)
                except Exception as e:
                    logger.warning(f"after_save hook {hook.__name__} failed: {e}", exc_info=False)

    def execute_before_connect(self, source: NodeT, rel_type: str, target: NodeT) -> None:
        """Execute before-connect hooks. First failure aborts.

        Hooks execute serially under lock for thread safety.
        """
        with self._lock:
            for hook in self._before_connect:
                hook(source, rel_type, target)  # Let exceptions propagate

    def execute_after_connect(self, source: NodeT, rel_type: str, target: NodeT) -> None:
        """Execute after-connect hooks. Failures are logged.

        Hooks execute serially under lock for thread safety.
        """
        with self._lock:
            for hook in self._after_connect:
                try:
                    hook(source, rel_type, target)
                except Exception as e:
                    logger.warning(f"after_connect hook {hook.__name__} failed: {e}", exc_info=False)

    def clear(self) -> None:
        """Clear all hooks. Mainly for testing."""
        with self._lock:
            self._before_save.clear()
            self._after_save.clear()
            self._before_connect.clear()
            self._after_connect.clear()


# Global singleton instance
_hooks = HookManager()


def get_hooks() -> HookManager:
    """Get the global hook manager."""
    return _hooks
