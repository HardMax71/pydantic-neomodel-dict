from threading import RLock
from typing import Any, Callable, Dict, Tuple, Type, TypeVar

from neomodel import StructuredNode
from neomodel.async_.core import AsyncStructuredNode
from pydantic import BaseModel

S = TypeVar("S")
T = TypeVar("T")


class ModelRegistry:
    """Thread-safe registry for model mappings and type converters.

    This is a singleton registry. All mappings are global across the application.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._pydantic_to_ogm: Dict[Type[BaseModel], Type[StructuredNode | AsyncStructuredNode]] = {}
        self._ogm_to_pydantic: Dict[Type[StructuredNode | AsyncStructuredNode], Type[BaseModel]] = {}
        self._type_converters: Dict[Tuple[Type, Type], Callable[[Any], Any]] = {}

    def register_models(
        self,
        pydantic_class: Type[BaseModel],
        ogm_class: Type[StructuredNode | AsyncStructuredNode]
    ) -> None:
        """Register bidirectional mapping between Pydantic and OGM models."""
        with self._lock:
            self._pydantic_to_ogm[pydantic_class] = ogm_class
            self._ogm_to_pydantic[ogm_class] = pydantic_class

    def get_ogm_class(self, pydantic_class: Type[BaseModel]) -> Type[StructuredNode | AsyncStructuredNode]:
        """Get OGM class for Pydantic class. Raises ConversionError if not registered."""
        from ..errors import ConversionError
        with self._lock:
            if pydantic_class not in self._pydantic_to_ogm:
                raise ConversionError(f"No mapping registered for Pydantic class {pydantic_class.__name__}")
            return self._pydantic_to_ogm[pydantic_class]

    def get_pydantic_class(self, ogm_class: Type[StructuredNode | AsyncStructuredNode]) -> Type[BaseModel]:
        """Get Pydantic class for OGM class. Raises ConversionError if not registered."""
        from ..errors import ConversionError
        with self._lock:
            if ogm_class not in self._ogm_to_pydantic:
                raise ConversionError(f"No Pydantic model registered for OGM class {ogm_class.__name__}. "
                                      f"No mapping registered for OGM class {ogm_class.__name__}")
            return self._ogm_to_pydantic[ogm_class]

    def register_type_converter(
        self,
        source_type: Type[S],
        target_type: Type[T],
        converter: Callable[[S], T]
    ) -> None:
        """Register custom type converter."""
        with self._lock:
            self._type_converters[(source_type, target_type)] = converter

    def convert_value(self, value: S | None, target_type: Type[T]) -> T | S | None:
        """Apply registered type converter if available, otherwise return value as-is."""
        if value is None:
            return None

        with self._lock:
            converter = self._type_converters.get((type(value), target_type))  # type: ignore

        if converter:
            return converter(value)  # type: ignore
        return value

    def clear(self) -> None:
        """Clear all registrations. Mainly for testing."""
        with self._lock:
            self._pydantic_to_ogm.clear()
            self._ogm_to_pydantic.clear()
            self._type_converters.clear()


# Global singleton instance
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _registry
