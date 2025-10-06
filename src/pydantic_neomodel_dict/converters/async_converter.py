"""Asynchronous converter for Pydantic ↔ OGM ↔ Dict."""

import logging
from typing import Any, Dict, List, Optional, Set, Union, get_args, get_origin

from neomodel.async_.core import AsyncStructuredNode
from pydantic import BaseModel

from ..core.registry import get_registry
from ..errors import ConversionError
from ..io import async_io

logger = logging.getLogger(__name__)


class AsyncConverter:
    """Asynchronous converter between Pydantic models, OGM nodes, and dictionaries."""

    async def to_ogm(
        self,
        pydantic_instance: BaseModel,
        ogm_class: Optional[type[AsyncStructuredNode]] = None,
        max_depth: int = 10
    ) -> AsyncStructuredNode:
        """Convert Pydantic instance to OGM node."""
        async with async_io.transaction():
            processed: Dict[int, AsyncStructuredNode] = {}
            return await self._to_ogm_recursive(
                pydantic_instance,
                ogm_class,
                processed,
                max_depth
            )

    async def batch_to_ogm(
        self,
        pydantic_instances: List[BaseModel],
        ogm_class: Optional[type[AsyncStructuredNode]] = None,
        max_depth: int = 10
    ) -> List[AsyncStructuredNode]:
        """Convert multiple Pydantic instances to OGM in one transaction."""
        async with async_io.transaction():
            processed: Dict[int, AsyncStructuredNode] = {}
            results = []
            for instance in pydantic_instances:
                node = await self._to_ogm_recursive(instance, ogm_class, processed, max_depth)
                results.append(node)
            return results

    async def batch_to_pydantic(
        self,
        ogm_instances: List[AsyncStructuredNode],
        pydantic_class: Optional[type[BaseModel]] = None,
        max_depth: int = 10
    ) -> List[BaseModel]:
        """Convert multiple OGM nodes to Pydantic models."""
        results = []
        for instance in ogm_instances:
            pydantic = await self.to_pydantic(instance, pydantic_class, max_depth)
            results.append(pydantic)
        return results

    async def batch_dict_to_ogm(
        self,
        data_dicts: List[Dict[str, Any]],
        ogm_class: type[AsyncStructuredNode],
        max_depth: int = 10
    ) -> List[AsyncStructuredNode]:
        """Convert multiple dicts to OGM nodes in one transaction."""
        async with async_io.transaction():
            processed: Dict[int, AsyncStructuredNode] = {}
            results = []
            for data in data_dicts:
                node = await self._dict_to_ogm_recursive(data, ogm_class, processed, max_depth)
                results.append(node)
            return results

    async def batch_ogm_to_dict(
        self,
        ogm_instances: List[AsyncStructuredNode],
        max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """Convert multiple OGM nodes to dicts."""
        results = []
        for instance in ogm_instances:
            result = await self.ogm_to_dict(instance, max_depth)
            results.append(result)
        return results

    async def _to_ogm_recursive(
        self,
        pydantic_instance: BaseModel,
        ogm_class: Optional[type[AsyncStructuredNode]],
        processed: Dict,  # Mixed keys: int for circular refs, tuple for deduplication
        max_depth: int
    ) -> AsyncStructuredNode:
        """Recursive implementation of to_ogm.

        Uses two-level caching:
        1. Object ID -> node (for circular reference detection)
        2. (ogm_class, unique_props) -> node (for deduplication by unique properties)
        """
        obj_id = id(pydantic_instance)
        if obj_id in processed:
            return processed[obj_id]

        if max_depth < 0:
            raise ConversionError(f"Max depth exceeded for {type(pydantic_instance).__name__}")

        if ogm_class is None:
            registry = get_registry()
            ogm_class = registry.get_ogm_class(type(pydantic_instance))

        data = self._extract_pydantic_properties(pydantic_instance)

        defined_props = ogm_class.defined_properties(rels=False, aliases=False)
        filtered_props = {
            k: v for k, v in data.items()
            if k in defined_props and v is not None
        }
        unique_props = {
            k: v for k, v in filtered_props.items()
            if defined_props[k].unique_index
        }

        if unique_props:
            cache_key = (ogm_class, tuple(sorted(unique_props.items())))
            if cache_key in processed:
                node = processed[cache_key]
                processed[obj_id] = node
                return node

        ogm_instance = await self._get_or_create_node(data, ogm_class)

        processed[obj_id] = ogm_instance

        if unique_props:
            processed[cache_key] = ogm_instance

        if max_depth > 0:
            await self._sync_relationships_from_pydantic(
                pydantic_instance,
                ogm_instance,
                processed,
                max_depth - 1
            )

        return ogm_instance

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

    async def _get_or_create_node(
        self,
        properties: Dict[str, Any],
        ogm_class: type[AsyncStructuredNode]
    ) -> AsyncStructuredNode:
        """Get or create node, with atomic upsert for unique properties."""
        defined_props = ogm_class.defined_properties(rels=False, aliases=False)

        # Filter to defined properties with non-None values
        filtered_props = {
            k: v for k, v in properties.items()
            if k in defined_props and v is not None
        }

        # Find unique properties
        unique_props = {
            k: v for k, v in filtered_props.items()
            if defined_props[k].unique_index
        }

        # If has unique properties: atomic merge on unique, set all
        if unique_props:
            return await async_io.merge_node_on_unique(ogm_class, unique_props, filtered_props)

        # No unique properties: create new
        node = ogm_class(**filtered_props)
        await async_io.save_node(node)
        return node

    async def _update_node_properties(
        self,
        node: AsyncStructuredNode,
        properties: Dict[str, Any]
    ) -> None:
        """Update node properties from dict."""
        defined = node.defined_properties(rels=False, aliases=False)
        registry = get_registry()

        for key, value in properties.items():
            if key in defined:
                prop_type = type(getattr(node, key, None))
                converted = registry.convert_value(value, prop_type)
                setattr(node, key, converted)

    async def _sync_relationships_from_pydantic(
        self,
        pydantic_instance: BaseModel,
        ogm_instance: AsyncStructuredNode,
        processed: Dict[int, AsyncStructuredNode],
        max_depth: int
    ) -> None:
        """Synchronize relationships from Pydantic to OGM."""
        ogm_class = type(ogm_instance)
        pydantic_class = type(pydantic_instance)

        ogm_rels = ogm_class.defined_properties(aliases=False, rels=True, properties=False)
        pydantic_fields = pydantic_class.model_fields

        common_rels = sorted(set(ogm_rels.keys()) & set(pydantic_fields.keys()))

        for rel_name in common_rels:
            rel_value = getattr(pydantic_instance, rel_name)

            if rel_value is None:
                continue

            rel_manager = getattr(ogm_instance, rel_name)
            rel_definition = ogm_rels[rel_name].definition
            target_ogm_class = rel_definition['node_class']

            is_list = rel_value.__class__ is list
            items = rel_value if is_list else [rel_value]

            related_nodes: List[AsyncStructuredNode] = []
            for item in items:
                related_node = await self._to_ogm_recursive(
                    item,
                    target_ogm_class,
                    processed,
                    max_depth
                )
                related_nodes.append(related_node)

            await self._sync_relationship(rel_manager, related_nodes)

    async def _sync_relationship(
        self,
        rel_manager,
        target_nodes: List[AsyncStructuredNode]
    ) -> None:
        """Synchronize relationship to match target nodes exactly."""
        existing = await async_io.get_all_related(rel_manager)
        existing_ids = {node.element_id for node in existing}
        target_ids = {node.element_id for node in target_nodes}

        to_add = target_ids - existing_ids
        to_remove = existing_ids - target_ids

        nodes_by_id = {node.element_id: node for node in target_nodes}
        for node_id in to_add:
            await async_io.connect_nodes(rel_manager, nodes_by_id[node_id])

        existing_by_id = {node.element_id: node for node in existing}
        for node_id in to_remove:
            await async_io.disconnect_nodes(rel_manager, existing_by_id[node_id])

    async def to_pydantic(
        self,
        ogm_instance: AsyncStructuredNode,
        pydantic_class: Optional[type[BaseModel]] = None,
        max_depth: int = 10
    ) -> BaseModel:
        """Convert OGM node to Pydantic model."""
        processed: Dict[str, BaseModel] = {}
        path: Set[str] = set()
        return await self._to_pydantic_recursive(
            ogm_instance,
            pydantic_class,
            processed,
            path,
            max_depth
        )

    async def _to_pydantic_recursive(
        self,
        ogm_instance: AsyncStructuredNode,
        pydantic_class: Optional[type[BaseModel]],
        processed: Dict[str, BaseModel],
        path: Set[str],
        max_depth: int
    ) -> BaseModel:
        """Recursive implementation of to_pydantic."""
        if not ogm_instance.element_id:
            await async_io.save_node(ogm_instance)

        element_id = ogm_instance.element_id

        # Check if already processed (before max_depth check)
        if element_id in processed and element_id not in path:
            return processed[element_id]

        if max_depth <= 0:
            return None

        if pydantic_class is None:
            registry = get_registry()
            pydantic_class = registry.get_pydantic_class(type(ogm_instance))

        if element_id in path:
            return self._create_minimal_pydantic(ogm_instance, pydantic_class)

        data = await self._extract_ogm_properties(ogm_instance, pydantic_class)
        pydantic_instance = pydantic_class.model_construct(**data)
        processed[element_id] = pydantic_instance

        path.add(element_id)

        try:
            if max_depth > 0:
                await self._load_pydantic_relationships(
                    ogm_instance,
                    pydantic_instance,
                    pydantic_class,
                    processed,
                    path,
                    max_depth - 1
                )
        finally:
            path.remove(element_id)

        return pydantic_instance

    def _create_minimal_pydantic(
        self,
        ogm_instance: AsyncStructuredNode,
        pydantic_class: type[BaseModel]
    ) -> BaseModel:
        """Create minimal Pydantic instance with all non-None property values.

        This ensures cycles get all available data, not just required/unique fields.
        """
        defined_props = ogm_instance.defined_properties(rels=False, aliases=False)
        pydantic_fields = pydantic_class.model_fields

        data = {}
        for prop_name, prop in defined_props.items():
            # Include property if it exists in Pydantic model and has a non-None value
            if prop_name in pydantic_fields:
                value = getattr(ogm_instance, prop_name)
                if value is not None:
                    data[prop_name] = value

        return pydantic_class.model_construct(**data)

    async def _extract_ogm_properties(
        self,
        ogm_instance: AsyncStructuredNode,
        pydantic_class: type[BaseModel]
    ) -> Dict[str, Any]:
        """Extract property values from OGM node."""
        defined_props = ogm_instance.defined_properties(rels=False, aliases=False)
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
            converted = registry.convert_value(value, field_type)
            data[prop_name] = converted

        return data

    async def _load_pydantic_relationships(
        self,
        ogm_instance: AsyncStructuredNode,
        pydantic_instance: BaseModel,
        pydantic_class: type[BaseModel],
        processed: Dict[str, BaseModel],
        path: Set[str],
        max_depth: int
    ) -> None:
        """Load relationships into Pydantic instance."""
        ogm_rels = ogm_instance.defined_properties(aliases=False, rels=True, properties=False)
        pydantic_fields = pydantic_class.model_fields
        registry = get_registry()

        for rel_name, rel in ogm_rels.items():
            if rel_name not in pydantic_fields:
                continue

            target_ogm_class = rel.definition['node_class']
            target_pydantic_class = registry.get_pydantic_class(target_ogm_class)

            rel_manager = getattr(ogm_instance, rel_name)
            related_nodes = await async_io.get_all_related(rel_manager)

            converted = []
            for node in related_nodes:
                pydantic_node = await self._to_pydantic_recursive(
                    node,
                    target_pydantic_class,
                    processed,
                    path,
                    max_depth
                )
                converted.append(pydantic_node)

            field_type = pydantic_fields[rel_name].annotation
            from typing import get_origin
            is_list = (
                get_origin(field_type) is list or
                field_type is list or
                (field_type.__class__.__name__ == '_GenericAlias' and
                 field_type.__origin__ is list)
            )

            if is_list:
                setattr(pydantic_instance, rel_name, converted)
            else:
                setattr(pydantic_instance, rel_name, converted[0] if converted else None)

    async def dict_to_ogm(
        self,
        data: Dict[str, Any],
        ogm_class: type[AsyncStructuredNode],
        max_depth: int = 10
    ) -> AsyncStructuredNode:
        """Convert dictionary to OGM node."""
        async with async_io.transaction():
            processed: Dict[int, AsyncStructuredNode] = {}
            return await self._dict_to_ogm_recursive(data, ogm_class, processed, max_depth)

    async def _dict_to_ogm_recursive(
        self,
        data: Dict[str, Any],
        ogm_class: type[AsyncStructuredNode],
        processed: Dict,  # Mixed keys: int for circular refs, tuple for deduplication
        max_depth: int
    ) -> AsyncStructuredNode:
        """Recursive implementation of dict_to_ogm."""
        if not isinstance(data, dict):
            raise ConversionError(
                f"Expected dict for {ogm_class.__name__}, got {type(data).__name__}"
            )

        data_id = id(data)
        if data_id in processed:
            return processed[data_id]

        if max_depth < 0:
            raise ConversionError(f"Max depth exceeded for {ogm_class.__name__}")

        defined_props = ogm_class.defined_properties(rels=False, aliases=False)
        defined_rels = ogm_class.defined_properties(aliases=False, rels=True, properties=False)

        properties = {k: v for k, v in data.items() if k in defined_props}
        relationships = {k: v for k, v in data.items() if k in defined_rels}

        filtered_props = {k: v for k, v in properties.items() if v is not None}
        unique_props = {
            k: v for k, v in filtered_props.items()
            if defined_props[k].unique_index
        }

        if unique_props:
            cache_key = (ogm_class, tuple(sorted(unique_props.items())))
            if cache_key in processed:
                node = processed[cache_key]
                processed[data_id] = node
                return node

        node = await self._get_or_create_node(properties, ogm_class)

        processed[data_id] = node

        if unique_props:
            processed[cache_key] = node

        if max_depth > 0:
            for rel_name, rel_data in relationships.items():
                if rel_data is None:
                    continue

                rel_def = defined_rels[rel_name].definition
                target_class = rel_def['node_class']
                rel_manager = getattr(node, rel_name)

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

                related_nodes = []
                for item in items:
                    if not isinstance(item, dict):
                        raise ConversionError(
                            f"Relationship '{rel_name}' must be a dictionary or list of dictionaries, "
                            f"got list item of type {type(item).__name__}"
                        )

                    related_node = await self._dict_to_ogm_recursive(
                        item,
                        target_class,
                        processed,
                        max_depth - 1
                    )
                    related_nodes.append(related_node)

                await self._sync_relationship(rel_manager, related_nodes)

        return node

    async def ogm_to_dict(
        self,
        ogm_instance: AsyncStructuredNode,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """Convert OGM node to dictionary."""
        processed: Dict[str, Dict[str, Any]] = {}
        path: Set[str] = set()
        return await self._ogm_to_dict_recursive(ogm_instance, processed, path, max_depth)

    async def _ogm_to_dict_recursive(
        self,
        ogm_instance: AsyncStructuredNode,
        processed: Dict[str, Dict[str, Any]],
        path: Set[str],
        max_depth: int
    ) -> Dict[str, Any]:
        """Recursive implementation of ogm_to_dict."""
        if not ogm_instance.element_id:
            await async_io.save_node(ogm_instance)

        element_id = ogm_instance.element_id

        # Check if already processed (before max_depth check)
        if element_id in processed and element_id not in path:
            return processed[element_id]

        if element_id in path:
            return self._extract_ogm_properties_as_dict(ogm_instance)

        if max_depth <= 0:
            return self._extract_ogm_properties_as_dict(ogm_instance)

        result = self._extract_ogm_properties_as_dict(ogm_instance)
        processed[element_id] = result

        path.add(element_id)

        try:
            ogm_rels = ogm_instance.defined_properties(aliases=False, rels=True, properties=False)

            for rel_name, rel in ogm_rels.items():
                rel_manager = getattr(ogm_instance, rel_name)
                related_nodes = await async_io.get_all_related(rel_manager)

                converted = []
                for node in related_nodes:
                    node_dict = await self._ogm_to_dict_recursive(
                        node,
                        processed,
                        path,
                        max_depth - 1
                    )
                    converted.append(node_dict)

                # Check cardinality - unwrap One/ZeroOrOne or incoming relationships
                # The rel_manager itself is an instance of the cardinality class
                from neomodel.async_.cardinality import One, ZeroOrOne

                is_single_cardinality = isinstance(rel_manager, (One, ZeroOrOne))
                rel_definition = rel_manager.definition
                is_incoming = rel_definition.get('direction') == -1  # RelationshipFrom

                # Check if self-referencing (same source and target class)
                target_class = rel_definition.get('node_class')
                source_class = rel_manager.source.__class__
                is_self_referencing = (target_class == source_class or
                                      target_class == source_class.__name__)

                # For self-referencing: check if ALL nodes in this relationship have exactly 1 link
                # This indicates a linked-list/chain structure that should be unwrapped
                unwrap_self_ref = False
                if is_self_referencing and len(converted) == 1:
                    # Check if this specific target node also has exactly 1 outgoing link
                    target_node = converted[0]
                    if isinstance(target_node, dict):
                        target_rel_value = target_node.get(rel_name)
                        # Unwrap if target also has single item (or is terminal/cycle)
                        if target_rel_value is None or isinstance(target_rel_value, dict):
                            unwrap_self_ref = True

                # Unwrap single items if: single-cardinality OR incoming OR self-ref chain
                should_unwrap_single = is_single_cardinality or is_incoming or unwrap_self_ref

                if len(converted) == 1 and should_unwrap_single:
                    # Unwrap single item
                    result[rel_name] = converted[0]
                elif len(converted) == 0 and is_single_cardinality:
                    # Empty single-cardinality (One/ZeroOrOne) should be None
                    result[rel_name] = None
                else:
                    # Keep as list for ZeroOrMore/OneOrMore, incoming, or self-referencing
                    result[rel_name] = converted
        finally:
            path.remove(element_id)

        return result

    def _extract_ogm_properties_as_dict(self, ogm_instance: AsyncStructuredNode) -> Dict[str, Any]:
        """Extract properties from OGM node as plain dict."""
        defined_props = ogm_instance.defined_properties(rels=False, aliases=False)

        result = {}
        for prop_name in defined_props.keys():
            value = getattr(ogm_instance, prop_name)
            if value is not None:
                result[prop_name] = value

        return result
