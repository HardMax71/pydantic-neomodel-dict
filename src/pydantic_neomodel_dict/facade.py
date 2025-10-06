from typing import Callable, Dict, List, Optional, Type, TypeVar

from neomodel import StructuredNode
from neomodel.async_.core import AsyncStructuredNode
from pydantic import BaseModel

from .converters import AsyncConverter as AsyncConverterImpl
from .converters import SyncConverter
from .core.hooks import get_hooks
from .core.registry import get_registry

S = TypeVar("S")
T = TypeVar("T")


class Converter:
    """Synchronous converter facade - maintains backwards compatibility."""

    _instance = SyncConverter()

    @classmethod
    def register_models(
        cls,
        pydantic_class: Type[BaseModel],
        ogm_class: Type[StructuredNode | AsyncStructuredNode]
    ) -> None:
        """Register bidirectional mapping between Pydantic and OGM models."""
        get_registry().register_models(pydantic_class, ogm_class)

    @classmethod
    def register_type_converter(
        cls,
        source_type: Type[S],
        target_type: Type[T],
        converter_func: Callable[[S], T]
    ) -> None:
        """Register custom type converter."""
        get_registry().register_type_converter(source_type, target_type, converter_func)

    @classmethod
    def register_before_save_hook(cls, hook: Callable) -> None:
        """Register before-save hook."""
        get_hooks().register_before_save(hook)

    @classmethod
    def register_after_save_hook(cls, hook: Callable) -> None:
        """Register after-save hook."""
        get_hooks().register_after_save(hook)

    @classmethod
    def register_before_connect_hook(cls, hook: Callable) -> None:
        """Register before-connect hook."""
        get_hooks().register_before_connect(hook)

    @classmethod
    def register_after_connect_hook(cls, hook: Callable) -> None:
        """Register after-connect hook."""
        get_hooks().register_after_connect(hook)

    @classmethod
    def clear_hooks(cls) -> None:
        """Clear all hooks."""
        get_hooks().clear()

    @classmethod
    def to_ogm(
        cls,
        pydantic_instance: BaseModel,
        ogm_class: Optional[Type[StructuredNode]] = None,
        processed_objects: Optional[Dict] = None,
        max_depth: int = 10,
        process_relationships: bool = True
    ) -> Optional[StructuredNode]:
        """Convert Pydantic to OGM.

        Note: processed_objects parameter is ignored in new implementation.
        Transaction handling is automatic.
        """
        return cls._instance.to_ogm(pydantic_instance, ogm_class, max_depth)

    @classmethod
    def to_pydantic(
        cls,
        ogm_instance: StructuredNode,
        pydantic_class: Optional[Type[BaseModel]] = None,
        processed_objects: Optional[Dict] = None,
        max_depth: int = 10,
        current_path: Optional[set] = None,
        include_relationships: bool = True,
        relationship_names: Optional[List[str]] = None
    ) -> Optional[BaseModel]:
        """Convert OGM to Pydantic.

        Note: Some parameters are ignored in new implementation for simplicity.
        """
        return cls._instance.to_pydantic(ogm_instance, pydantic_class, max_depth)

    @classmethod
    def dict_to_ogm(
        cls,
        data_dict: dict,
        ogm_class: Type[StructuredNode],
        processed_objects: Optional[Dict] = None,
        max_depth: int = 10,
        process_relationships: bool = True
    ) -> Optional[StructuredNode]:
        """Convert dict to OGM."""
        return cls._instance.dict_to_ogm(data_dict, ogm_class, max_depth)

    @classmethod
    def ogm_to_dict(
        cls,
        ogm_instance: StructuredNode,
        processed_objects: Optional[Dict] = None,
        max_depth: int = 10,
        current_path: Optional[set] = None,
        include_properties: bool = True,
        include_relationships: bool = True,
        relationship_names: Optional[List[str]] = None
    ) -> Optional[dict]:
        """Convert OGM to dict."""
        return cls._instance.ogm_to_dict(
            ogm_instance, max_depth, include_properties, include_relationships
        )

    @classmethod
    def batch_to_ogm(
        cls,
        pydantic_instances: List[BaseModel],
        ogm_class: Optional[Type[StructuredNode]] = None,
        max_depth: int = 10
    ) -> List[StructuredNode]:
        """Batch convert Pydantic to OGM in single transaction."""
        return cls._instance.batch_to_ogm(pydantic_instances, ogm_class, max_depth)

    @classmethod
    def batch_to_pydantic(
        cls,
        ogm_instances: List[StructuredNode],
        pydantic_class: Optional[Type[BaseModel]] = None,
        max_depth: int = 10
    ) -> List[BaseModel]:
        """Batch convert OGM to Pydantic."""
        return cls._instance.batch_to_pydantic(ogm_instances, pydantic_class, max_depth)

    @classmethod
    def batch_dict_to_ogm(
        cls,
        data_dicts: List[dict],
        ogm_class: Type[StructuredNode],
        max_depth: int = 10
    ) -> List[StructuredNode]:
        """Batch convert dicts to OGM in single transaction."""
        return cls._instance.batch_dict_to_ogm(data_dicts, ogm_class, max_depth)

    @classmethod
    def batch_ogm_to_dict(
        cls,
        ogm_instances: List[StructuredNode],
        max_depth: int = 10
    ) -> List[dict]:
        """Batch convert OGM to dict."""
        return cls._instance.batch_ogm_to_dict(ogm_instances, max_depth)

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all model registrations. Mainly for testing."""
        get_registry().clear()

# Backwards compatibility: expose registry internals as class attributes
Converter._pydantic_to_ogm = get_registry()._pydantic_to_ogm
Converter._ogm_to_pydantic = get_registry()._ogm_to_pydantic



class AsyncConverterFacade:
    """Asynchronous converter facade - maintains backwards compatibility."""

    _instance = AsyncConverterImpl()

    @classmethod
    def register_models(
        cls,
        pydantic_class: Type[BaseModel],
        ogm_class: Type[AsyncStructuredNode]
    ) -> None:
        """Register bidirectional mapping."""
        get_registry().register_models(pydantic_class, ogm_class)

    @classmethod
    async def to_ogm(
        cls,
        pydantic_instance: BaseModel,
        ogm_class: Optional[Type[AsyncStructuredNode]] = None,
        processed_objects: Optional[Dict] = None,
        max_depth: int = 10
    ) -> Optional[AsyncStructuredNode]:
        """Convert Pydantic to OGM."""
        return await cls._instance.to_ogm(pydantic_instance, ogm_class, max_depth)

    @classmethod
    async def to_pydantic(
        cls,
        ogm_instance: AsyncStructuredNode,
        pydantic_class: Optional[Type[BaseModel]] = None,
        processed_objects: Optional[Dict] = None,
        max_depth: int = 10,
        current_path: Optional[set] = None
    ) -> Optional[BaseModel]:
        """Convert OGM to Pydantic."""
        return await cls._instance.to_pydantic(ogm_instance, pydantic_class, max_depth)

    @classmethod
    async def dict_to_ogm(
        cls,
        data_dict: dict,
        ogm_class: Type[AsyncStructuredNode],
        processed_objects: Optional[Dict] = None,
        max_depth: int = 10
    ) -> Optional[AsyncStructuredNode]:
        """Convert dict to OGM."""
        return await cls._instance.dict_to_ogm(data_dict, ogm_class, max_depth)

    @classmethod
    async def ogm_to_dict(
        cls,
        ogm_instance: AsyncStructuredNode,
        processed_objects: Optional[Dict] = None,
        max_depth: int = 10,
        current_path: Optional[set] = None
    ) -> Optional[dict]:
        """Convert OGM to dict."""
        return await cls._instance.ogm_to_dict(ogm_instance, max_depth)

    @classmethod
    async def batch_to_ogm(
        cls,
        pydantic_instances: List[BaseModel],
        ogm_class: Optional[Type[AsyncStructuredNode]] = None,
        max_depth: int = 10
    ) -> List[AsyncStructuredNode]:
        """Batch convert Pydantic to OGM in single transaction."""
        return await cls._instance.batch_to_ogm(pydantic_instances, ogm_class, max_depth)

    @classmethod
    async def batch_to_pydantic(
        cls,
        ogm_instances: List[AsyncStructuredNode],
        pydantic_class: Optional[Type[BaseModel]] = None,
        max_depth: int = 10
    ) -> List[BaseModel]:
        """Batch convert OGM to Pydantic."""
        return await cls._instance.batch_to_pydantic(ogm_instances, pydantic_class, max_depth)

    @classmethod
    async def batch_dict_to_ogm(
        cls,
        data_dicts: List[dict],
        ogm_class: Type[AsyncStructuredNode],
        max_depth: int = 10
    ) -> List[AsyncStructuredNode]:
        """Batch convert dicts to OGM in single transaction."""
        return await cls._instance.batch_dict_to_ogm(data_dicts, ogm_class, max_depth)

    @classmethod
    async def batch_ogm_to_dict(
        cls,
        ogm_instances: List[AsyncStructuredNode],
        max_depth: int = 10
    ) -> List[dict]:
        """Batch convert OGM to dict."""
        return await cls._instance.batch_ogm_to_dict(ogm_instances, max_depth)
