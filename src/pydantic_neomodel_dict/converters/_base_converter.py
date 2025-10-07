from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar, Union, get_args, get_origin

from neomodel import RelationshipManager, StructuredNode
from neomodel.async_.core import AsyncStructuredNode
from neomodel.async_.relationship_manager import AsyncRelationshipManager
from pydantic import BaseModel

from ..core.registry import get_registry
from ..errors import ConversionError

NodeT = TypeVar('NodeT', StructuredNode, AsyncStructuredNode)
RelManagerT = TypeVar('RelManagerT', RelationshipManager, AsyncRelationshipManager)
S = TypeVar("S")
T = TypeVar("T")


class BaseConverter(ABC, Generic[NodeT, RelManagerT]):
    """Abstract base converter with shared logic between sync and async implementations.

    All I/O operations are abstract and must be implemented by subclasses.
    All non-I/O logic is shared and implemented here.
    """

    @abstractmethod
    def _save_node(self, node: NodeT) -> Any:
        """Save node to database (sync or async)."""

    @abstractmethod
    def _get_all_related(self, rel_manager: RelManagerT) -> Any:
        """Get all related nodes (sync or async)."""

    @abstractmethod
    def _connect_nodes(self, rel_manager: RelManagerT, target: NodeT) -> Any:
        """Connect nodes (sync or async)."""

    @abstractmethod
    def _disconnect_nodes(self, rel_manager: RelManagerT, target: NodeT) -> Any:
        """Disconnect nodes (sync or async)."""

    @abstractmethod
    def _merge_node_on_unique(
            self,
            ogm_class: type[NodeT],
            unique_props: Dict[str, Any],
            all_props: Dict[str, Any]
    ) -> Any:
        """Merge node with unique properties (sync or async)."""

    @abstractmethod
    def _transaction(self) -> Any:
        """Transaction context manager (sync or async)."""

    # ----- Registry helpers -----
    def register_models(self, pydantic_class: Type[BaseModel], ogm_class: Type[NodeT]) -> None:
        get_registry().register_models(pydantic_class, ogm_class)

    def register_type_converter(self, source_type: Type[S], target_type: Type[T],
                                converter_func: Callable[[S], T]) -> None:
        get_registry().register_type_converter(source_type, target_type, converter_func)

    def clear_registry(self) -> None:
        get_registry().clear()

    def _extract_pydantic_properties(self, pydantic_instance: BaseModel) -> Dict[str, Any]:
        """Extract property values from Pydantic instance.

        Only extracts simple properties, not nested BaseModel instances.
        Uses direct field access to avoid circular reference issues in model_dump().
        """
        cleaned = {}

        model_fields = type(pydantic_instance).model_fields

        for field_name, field_info in model_fields.items():
            field_type = field_info.annotation

            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    field_type = non_none_args[0]
                    origin = get_origin(field_type)

            try:
                if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                    continue
            except TypeError:
                pass

            if origin is list:
                args = get_args(field_type)
                if args:
                    try:
                        if isinstance(args[0], type) and issubclass(args[0], BaseModel):
                            continue
                    except TypeError:
                        pass

            if hasattr(pydantic_instance, field_name):
                value = getattr(pydantic_instance, field_name)
                if value is not None:
                    cleaned[field_name] = value

        return cleaned

    def _create_minimal_pydantic(
            self,
            ogm_instance: NodeT,
            pydantic_class: type[BaseModel]
    ) -> BaseModel:
        """Create minimal Pydantic instance with all non-None property values.

        This ensures cycles get all available data, not just required/unique fields.
        """
        defined_props = ogm_instance.defined_properties(rels=False, aliases=False)  # type: ignore[attr-defined]
        pydantic_fields = pydantic_class.model_fields

        data = {}
        for prop_name, prop in defined_props.items():
            if prop_name in pydantic_fields:
                value = getattr(ogm_instance, prop_name)
                if value is not None:
                    data[prop_name] = value

        return pydantic_class.model_construct(**data)

    def _is_list_annotation(self, field_type: Any) -> bool:
        """Detect whether a Pydantic field annotation represents a list.

        Handles typing constructs like list[T], List[T], Optional[List[T]] (Union), and
        runtime generic aliases used by some Python versions.
        """
        origin = get_origin(field_type)

        # Optional[List[T]] or Union[List[T], None]
        if origin is Union:
            args = [a for a in get_args(field_type) if a is not type(None)]
            if len(args) == 1:
                return self._is_list_annotation(args[0])

        if origin is list:
            return True

        if field_type is list:
            return True

        # Fallback for runtime generic aliases: field_type.__origin__ may exist
        try:
            return getattr(field_type, "__origin__", None) is list
        except Exception:
            return False

    def _filter_defined_properties(
            self,
            ogm_class: type[NodeT],
            properties: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Filter dict to properties defined on the OGM class and split unique ones.

        Returns a tuple of (filtered_props, unique_props). Values with None are excluded.
        """
        defined_props = ogm_class.defined_properties(rels=False, aliases=False)  # type: ignore[attr-defined]

        filtered_props: Dict[str, Any] = {
            k: v for k, v in properties.items()
            if k in defined_props and v is not None
        }

        unique_props: Dict[str, Any] = {
            k: v for k, v in filtered_props.items()
            if defined_props[k].unique_index
        }

        return filtered_props, unique_props

    def _build_unique_cache_key(
            self,
            ogm_class: type[NodeT],
            unique_props: Dict[str, Any]
    ) -> Tuple[type[NodeT], Tuple[Tuple[str, Any], ...]]:
        """Build a stable cache key for unique properties deduplication."""
        return ogm_class, tuple(sorted(unique_props.items()))

    def _relationship_delta(
            self,
            existing_nodes: List[NodeT],
            target_nodes: List[NodeT]
    ) -> Tuple[Set[str], Set[str], Dict[str, NodeT], Dict[str, NodeT]]:
        """Compute differences between existing and target related nodes by element_id.

        Returns:
            to_add: set of element_ids to connect
            to_remove: set of element_ids to disconnect
            targets_by_id: mapping for quick lookup when connecting
            existing_by_id: mapping for quick lookup when disconnecting
        """
        existing_ids = {node.element_id for node in existing_nodes}
        target_ids = {node.element_id for node in target_nodes}

        to_add = target_ids - existing_ids
        to_remove = existing_ids - target_ids

        targets_by_id = {node.element_id: node for node in target_nodes}
        existing_by_id = {node.element_id: node for node in existing_nodes}

        return to_add, to_remove, targets_by_id, existing_by_id  # type: ignore[return-value]

    def _resolve_ogm_class_from_instance(
            self,
            pydantic_instance: BaseModel,
            ogm_class: Optional[type[NodeT]]
    ) -> type[NodeT]:
        if ogm_class is not None:
            return ogm_class
        registry = get_registry()
        return registry.get_ogm_class(type(pydantic_instance))  # type: ignore[return-value]

    def _to_ogm_prepare(
            self,
            pydantic_instance: BaseModel,
            ogm_class: Optional[type[NodeT]],
            processed: Dict[Any, Any],
            max_depth: int
    ) -> Tuple[
        Optional[NodeT],
        type[NodeT],
        int,
        Dict[str, Any],
        Dict[str, Any],
        Optional[Tuple[type[NodeT], Tuple[Tuple[str, Any], ...]]]
    ]:
        obj_id = id(pydantic_instance)
        if obj_id in processed:
            return processed[obj_id], None, obj_id, {}, {}, None  # type: ignore[return-value]

        if max_depth < 0:
            raise ConversionError(f"Max depth exceeded for {type(pydantic_instance).__name__}")

        resolved_ogm = self._resolve_ogm_class_from_instance(pydantic_instance, ogm_class)
        data = self._extract_pydantic_properties(pydantic_instance)
        filtered_props, unique_props = self._filter_defined_properties(resolved_ogm, data)

        cache_key = None
        if unique_props:
            cache_key = self._build_unique_cache_key(resolved_ogm, unique_props)
            if cache_key in processed:
                node = processed[cache_key]
                processed[obj_id] = node
                return node, resolved_ogm, obj_id, filtered_props, unique_props, cache_key

        return None, resolved_ogm, obj_id, filtered_props, unique_props, cache_key

    def _to_ogm_finalize(
            self,
            processed: Dict[Any, Any],
            obj_id: int,
            node: NodeT,
            unique_props: Dict[str, Any],
            cache_key: Optional[Tuple[type[NodeT], Tuple[Tuple[str, Any], ...]]]
    ) -> None:
        processed[obj_id] = node
        if unique_props and cache_key is not None:
            processed[cache_key] = node

    def _dict_to_ogm_prepare(
            self,
            data: Dict[str, Any],
            ogm_class: type[NodeT],
            processed: Dict[Any, Any],
            max_depth: int
    ) -> Tuple[
        Optional[NodeT],
        int,
        Dict[str, Any],
        Dict[str, Any],
        Optional[Tuple[type[NodeT], Tuple[Tuple[str, Any], ...]]],
        Dict[str, Any]
    ]:
        if not isinstance(data, dict):
            raise ConversionError(
                f"Expected dict for {ogm_class.__name__}, got {type(data).__name__}"
            )

        data_id = id(data)
        if data_id in processed:
            defined_rels = ogm_class.defined_properties(aliases=False, rels=True,
                                                        properties=False)  # type: ignore[attr-defined]
            return processed[data_id], data_id, {}, {}, None, defined_rels  # type: ignore[return-value]

        if max_depth < 0:
            raise ConversionError(f"Max depth exceeded for {ogm_class.__name__}")

        defined_props = ogm_class.defined_properties(rels=False, aliases=False)  # type: ignore[attr-defined]
        defined_rels = ogm_class.defined_properties(aliases=False, rels=True,
                                                    properties=False)  # type: ignore[attr-defined]

        properties = {k: v for k, v in data.items() if k in defined_props}
        relationships = {k: v for k, v in data.items() if k in defined_rels}

        filtered_props, unique_props = self._filter_defined_properties(ogm_class, properties)

        cache_key = None
        if unique_props:
            cache_key = self._build_unique_cache_key(ogm_class, unique_props)
            if cache_key in processed:
                node = processed[cache_key]
                processed[data_id] = node
                return node, data_id, properties, relationships, cache_key, defined_rels

        return None, data_id, properties, relationships, cache_key, defined_rels

    def _iter_pydantic_relationship_value(self, value: Any) -> List[Any]:
        """Normalize a relationship value from Pydantic to a list of items."""
        return value if (value.__class__ is list) else [value]

    def _enumerate_pydantic_relationships(
        self,
        pydantic_instance: BaseModel,
        ogm_instance: NodeT,
    ) -> List[Tuple[RelManagerT, type[NodeT], List[Any]]]:
        """Prepare relationship sync tasks from a Pydantic instance into an OGM instance.

        Returns a list of triples: (rel_manager, target_ogm_class, items)
        where items is a normalized list of related Pydantic models to convert.
        """
        ogm_rels: Dict[str, Any] = type(ogm_instance).defined_properties(aliases=False, rels=True, properties=False)  # type: ignore[attr-defined]
        pydantic_fields: Dict[str, Any] = type(pydantic_instance).model_fields

        common_rels = self._common_relationship_names(ogm_rels, pydantic_fields)

        tasks: List[Tuple[RelManagerT, type[NodeT], List[Any]]] = []
        for rel_name in common_rels:
            rel_value = getattr(pydantic_instance, rel_name)
            if rel_value is None:
                continue

            rel_manager: RelManagerT = getattr(ogm_instance, rel_name)
            rel_definition: Dict[str, Any] = ogm_rels[rel_name].definition
            target_ogm_class: type[NodeT] = rel_definition['node_class']

            items = self._iter_pydantic_relationship_value(rel_value)
            tasks.append((rel_manager, target_ogm_class, items))

        return tasks

    def _enumerate_ogm_relationship_targets(
        self,
        ogm_instance: NodeT,
        pydantic_class: type[BaseModel],
    ) -> List[Tuple[str, RelManagerT, type[BaseModel]]]:
        ogm_rels: Dict[str, Any] = ogm_instance.defined_properties(aliases=False, rels=True, properties=False)
        pydantic_fields: Dict[str, Any] = pydantic_class.model_fields
        registry = get_registry()

        tasks: List[Tuple[str, RelManagerT, type[BaseModel]]] = []
        for rel_name in self._common_relationship_names(ogm_rels, pydantic_fields):
            rel = ogm_rels[rel_name]
            target_ogm_class = rel.definition['node_class']
            target_pydantic_class = registry.get_pydantic_class(target_ogm_class)
            rel_manager: RelManagerT = getattr(ogm_instance, rel_name)
            tasks.append((rel_name, rel_manager, target_pydantic_class))

        return tasks

    def _iter_ogm_relationship_managers(
        self,
        ogm_instance: NodeT,
    ) -> List[Tuple[str, RelManagerT]]:
        ogm_rels: Dict[str, Any] = ogm_instance.defined_properties(aliases=False, rels=True, properties=False)  # type: ignore[attr-defined]
        return [
            (rel_name, getattr(ogm_instance, rel_name))
            for rel_name in ogm_rels.keys()
        ]

    def _prepare_ogm_to_dict(
        self,
        ogm_instance: NodeT,
        processed: Dict[str, Dict[str, Any]],
        path: Set[str],
        max_depth: int,
        include_properties: bool,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        element_id = ogm_instance.element_id
        assert element_id is not None

        stop, short = self._ogm_to_dict_prechecks(
            element_id, ogm_instance, processed, path, max_depth, include_properties
        )
        if stop:
            return False, element_id, short

        result: Dict[str, Any] = (
            self._extract_ogm_properties_as_dict(ogm_instance) if include_properties else {}
        )
        processed[element_id] = result
        return True, element_id, result

    def _normalize_relationship_input(self, rel_name: str, rel_data: Any) -> List[Dict[str, Any]]:
        is_list = rel_data.__class__ is list
        if is_list:
            items = rel_data
        elif isinstance(rel_data, dict):
            items = [rel_data]
        else:
            raise ConversionError(
                f"Relationship '{rel_name}' must be a dictionary or list of dictionaries, "
                f"got {type(rel_data).__name__}"
            )

        for item in items:
            if not isinstance(item, dict):
                raise ConversionError(
                    f"Relationship '{rel_name}' must be a dictionary or list of dictionaries, "
                    f"got list item of type {type(item).__name__}"
                )

        return items  # type: ignore[no-any-return]

    def _resolve_pydantic_class(
            self,
            ogm_instance: NodeT,
            pydantic_class: Optional[type[BaseModel]]
    ) -> type[BaseModel]:
        if pydantic_class is not None:
            return pydantic_class
        registry = get_registry()
        return registry.get_pydantic_class(type(ogm_instance))

    def _to_pydantic_prechecks(
            self,
            element_id: str,
            pydantic_class: type[BaseModel],
            ogm_instance: NodeT,
            processed: Dict[str, BaseModel],
            path: Set[str],
            max_depth: int
    ) -> Tuple[bool, Optional[BaseModel]]:
        # If already processed (and not currently in path), return cached
        if element_id in processed and element_id not in path:
            return True, processed[element_id]

        if max_depth <= 0:
            return True, None

        if element_id in path:
            return True, self._create_minimal_pydantic(ogm_instance, pydantic_class)

        return False, None

    def _ogm_to_dict_prechecks(
            self,
            element_id: str,
            ogm_instance: NodeT,
            processed: Dict[str, Dict[str, Any]],
            path: Set[str],
            max_depth: int,
            include_properties: bool
    ) -> Tuple[bool, Dict[str, Any]]:
        if element_id in processed and element_id not in path:
            return True, processed[element_id]

        if element_id in path:
            if include_properties:
                return True, self._extract_ogm_properties_as_dict(ogm_instance)
            return True, {}

        if max_depth <= 0:
            if include_properties:
                return True, self._extract_ogm_properties_as_dict(ogm_instance)
            return True, {}

        return False, {}

    def _common_relationship_names(
            self,
            ogm_rels: Dict[str, Any],
            pydantic_fields: Dict[str, Any]
    ) -> List[str]:
        """Return sorted intersection of relationship names present in both schemas."""
        return sorted(set(ogm_rels.keys()) & set(pydantic_fields.keys()))

    def _assign_relationship_value(
            self,
            pydantic_instance: BaseModel,
            rel_name: str,
            field_type: Any,
            converted: List[Any]
    ) -> None:
        """Assign relationship value on Pydantic instance based on field annotation.

        If field is list-like, assign the whole list; otherwise assign first or None.
        """
        is_list = self._is_list_annotation(field_type)
        if is_list:
            setattr(pydantic_instance, rel_name, converted)
        else:
            setattr(pydantic_instance, rel_name, converted[0] if converted else None)

    def _extract_ogm_properties(
            self,
            ogm_instance: NodeT,
            pydantic_class: type[BaseModel]
    ) -> Dict[str, Any]:
        """Extract property values from OGM node."""
        defined_props = ogm_instance.defined_properties(rels=False, aliases=False)  # type: ignore[attr-defined]
        pydantic_fields = pydantic_class.model_fields
        registry = get_registry()

        data = {}
        for prop_name in defined_props.keys():
            if prop_name not in pydantic_fields:
                continue

            value = getattr(ogm_instance, prop_name)
            if value is None:
                continue

            field_type = pydantic_fields[prop_name].annotation
            converted = registry.convert_value(value, field_type)  # type: ignore[arg-type]
            data[prop_name] = converted

        return data

    def _extract_ogm_properties_as_dict(self, ogm_instance: NodeT) -> Dict[str, Any]:
        """Extract properties from OGM node as plain dict.

        If a Pydantic model is registered for this OGM class, use its type hints
        to convert lists to sets where appropriate.
        """
        defined_props = ogm_instance.defined_properties(rels=False, aliases=False)  # type: ignore[attr-defined]

        result = {}
        for prop_name in defined_props.keys():
            value = getattr(ogm_instance, prop_name)
            if value is not None:
                result[prop_name] = value

        registry = get_registry()
        try:
            pydantic_class = registry.get_pydantic_class(type(ogm_instance))  # type: ignore[arg-type]
            model_fields = pydantic_class.model_fields

            for prop_name, value in result.items():
                if prop_name in model_fields and isinstance(value, list):
                    field_info = model_fields[prop_name]
                    field_type = field_info.annotation

                    origin = get_origin(field_type)
                    if origin is set:
                        result[prop_name] = set(value)
        except ConversionError:
            pass

        return result
