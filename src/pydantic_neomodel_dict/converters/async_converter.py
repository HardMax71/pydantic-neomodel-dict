from typing import Any, Dict, List, Optional, Set

from neomodel.async_.core import AsyncStructuredNode
from neomodel.async_.relationship_manager import AsyncRelationshipManager
from pydantic import BaseModel

from ..io import async_io
from ..io._common import unwrap_relationship_value
from ._base_converter import BaseConverter


class AsyncConverter(BaseConverter[AsyncStructuredNode, AsyncRelationshipManager]):
    """Asynchronous converter between Pydantic models, OGM nodes, and dictionaries."""

    async def _save_node(self, node: AsyncStructuredNode) -> None:
        await async_io.save_node(node)

    async def _get_all_related(self, rel_manager: AsyncRelationshipManager) -> List[AsyncStructuredNode]:
        return await async_io.get_all_related(rel_manager)

    async def _connect_nodes(self, rel_manager: AsyncRelationshipManager, target: AsyncStructuredNode) -> None:
        await async_io.connect_nodes(rel_manager, target)

    async def _disconnect_nodes(self, rel_manager: AsyncRelationshipManager, target: AsyncStructuredNode) -> None:
        await async_io.disconnect_nodes(rel_manager, target)

    async def _merge_node_on_unique(
            self,
            ogm_class: type[AsyncStructuredNode],
            unique_props: Dict[str, Any],
            all_props: Dict[str, Any]
    ) -> AsyncStructuredNode:
        return await async_io.merge_node_on_unique(ogm_class, unique_props, all_props)

    def _transaction(self):  # type: ignore[no-untyped-def]
        return async_io.transaction()

    async def to_ogm(
            self,
            pydantic_instance: BaseModel,
            ogm_class: Optional[type[AsyncStructuredNode]] = None,
            max_depth: int = 10
    ) -> AsyncStructuredNode:
        """Convert Pydantic instance to OGM node."""
        async with self._transaction():
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
        async with self._transaction():
            processed: Dict[int, AsyncStructuredNode] = {}
            return [
                await self._to_ogm_recursive(instance, ogm_class, processed, max_depth)
                for instance in pydantic_instances
            ]

    async def batch_to_pydantic(
            self,
            ogm_instances: List[AsyncStructuredNode],
            pydantic_class: Optional[type[BaseModel]] = None,
            max_depth: int = 10
    ) -> List[BaseModel]:
        """Convert multiple OGM nodes to Pydantic models."""
        return [
            await self.to_pydantic(instance, pydantic_class, max_depth)
            for instance in ogm_instances
        ]

    async def batch_dict_to_ogm(
            self,
            data_dicts: List[Dict[str, Any]],
            ogm_class: type[AsyncStructuredNode],
            max_depth: int = 10
    ) -> List[AsyncStructuredNode]:
        """Convert multiple dicts to OGM nodes in one transaction."""
        async with self._transaction():
            processed: Dict[int, AsyncStructuredNode] = {}
            return [
                await self._dict_to_ogm_recursive(data, ogm_class, processed, max_depth)
                for data in data_dicts
            ]

    async def batch_ogm_to_dict(
            self,
            ogm_instances: List[AsyncStructuredNode],
            max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """Convert multiple OGM nodes to dicts."""
        return [
            await self.ogm_to_dict(instance, max_depth)
            for instance in ogm_instances
        ]

    async def _to_ogm_recursive(
            self,
            pydantic_instance: BaseModel,
            ogm_class: Optional[type[AsyncStructuredNode]],
            processed: Dict,  # Mixed keys: int for circular refs, tuple for deduplication
            max_depth: int
    ) -> AsyncStructuredNode:
        """Recursive implementation of to_ogm using shared prep/finalize helpers."""
        existing, resolved_ogm, obj_id, filtered_props, unique_props, cache_key = self._to_ogm_prepare(
            pydantic_instance, ogm_class, processed, max_depth
        )

        if existing is not None:
            return existing

        ogm_instance = await self._get_or_create_node(filtered_props, resolved_ogm)
        self._to_ogm_finalize(processed, obj_id, ogm_instance, unique_props, cache_key)

        if max_depth > 0:
            await self._sync_relationships_from_pydantic(
                pydantic_instance,
                ogm_instance,
                processed,
                max_depth - 1
            )

        return ogm_instance

    async def _get_or_create_node(
            self,
            properties: Dict[str, Any],
            ogm_class: type[AsyncStructuredNode]
    ) -> AsyncStructuredNode:
        """Get or create node, with atomic upsert for unique properties."""
        filtered_props, unique_props = self._filter_defined_properties(ogm_class, properties)

        if unique_props:
            return await self._merge_node_on_unique(ogm_class, unique_props, filtered_props)

        node = ogm_class(**filtered_props)
        await self._save_node(node)
        return node

    async def _sync_relationships_from_pydantic(
            self,
            pydantic_instance: BaseModel,
            ogm_instance: AsyncStructuredNode,
            processed: Dict[int, AsyncStructuredNode],
            max_depth: int
    ) -> None:
        """Synchronize relationships from Pydantic to OGM."""
        for rel_manager, target_ogm_class, items in self._enumerate_pydantic_relationships(
            pydantic_instance, ogm_instance
        ):
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
            rel_manager: AsyncRelationshipManager,
            target_nodes: List[AsyncStructuredNode]
    ) -> None:
        """Synchronize relationship to match target nodes exactly."""
        existing = await self._get_all_related(rel_manager)
        to_add, to_remove, targets_by_id, existing_by_id = self._relationship_delta(existing, target_nodes)

        for node_id in to_add:
            await self._connect_nodes(rel_manager, targets_by_id[node_id])

        for node_id in to_remove:
            await self._disconnect_nodes(rel_manager, existing_by_id[node_id])

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
        """Recursive implementation of to_pydantic with shared prechecks."""
        if not ogm_instance.element_id:
            await self._save_node(ogm_instance)

        element_id = ogm_instance.element_id
        assert element_id is not None
        resolved = self._resolve_pydantic_class(ogm_instance, pydantic_class)
        stop, result = self._to_pydantic_prechecks(
            element_id, resolved, ogm_instance, processed, path, max_depth
        )
        if stop:
            return result  # type: ignore[return-value]

        data = self._extract_ogm_properties(ogm_instance, resolved)
        pydantic_instance = resolved.model_construct(**data)
        processed[element_id] = pydantic_instance

        path.add(element_id)

        try:
            await self._load_pydantic_relationships(
                ogm_instance,
                pydantic_instance,
                resolved,
                processed,
                path,
                max_depth - 1
            )
        finally:
            path.remove(element_id)

        return pydantic_instance

    async def _load_pydantic_relationships(
            self,
            ogm_instance: AsyncStructuredNode,
            pydantic_instance: BaseModel,
            pydantic_class: type[BaseModel],
            processed: Dict[str, BaseModel],
            path: Set[str],
            max_depth: int
    ) -> None:
        """Load relationships into Pydantic instance (no-op if max_depth <= 0)."""
        if max_depth <= 0:
            return
        pydantic_fields = pydantic_class.model_fields
        for rel_name, rel_manager, target_pydantic_class in self._enumerate_ogm_relationship_targets(
            ogm_instance, pydantic_class
        ):
            related_nodes = await self._get_all_related(rel_manager)
            converted = [
                await self._to_pydantic_recursive(
                    node,
                    target_pydantic_class,
                    processed,
                    path,
                    max_depth
                )
                for node in related_nodes
            ]
            field_type = pydantic_fields[rel_name].annotation
            self._assign_relationship_value(pydantic_instance, rel_name, field_type, converted)

    async def dict_to_ogm(
            self,
            data: Dict[str, Any],
            ogm_class: type[AsyncStructuredNode],
            max_depth: int = 10
    ) -> AsyncStructuredNode:
        """Convert dictionary to OGM node."""
        async with self._transaction():
            processed: Dict[int, AsyncStructuredNode] = {}
            return await self._dict_to_ogm_recursive(data, ogm_class, processed, max_depth)

    async def _dict_to_ogm_recursive(
            self,
            data: Dict[str, Any],
            ogm_class: type[AsyncStructuredNode],
            processed: Dict,  # Mixed keys: int for circular refs, tuple for deduplication
            max_depth: int
    ) -> AsyncStructuredNode:
        """Recursive implementation of dict_to_ogm using shared preparation."""
        existing, data_id, properties, relationships, cache_key, defined_rels = (
            self._dict_to_ogm_prepare(data, ogm_class, processed, max_depth)
        )

        if existing is not None:
            return existing

        node = await self._get_or_create_node(properties, ogm_class)
        processed[data_id] = node
        if cache_key is not None:
            processed[cache_key] = node

        if max_depth > 0:
            for rel_name, rel_data in relationships.items():
                if rel_data is None:
                    continue

                rel_def = defined_rels[rel_name].definition
                target_class = rel_def['node_class']
                rel_manager = getattr(node, rel_name)

                items = self._normalize_relationship_input(rel_name, rel_data)
                related_nodes: List[AsyncStructuredNode] = [
                    await self._dict_to_ogm_recursive(item, target_class, processed, max_depth - 1)
                    for item in items
                ]

                await self._sync_relationship(rel_manager, related_nodes)

        return node

    async def ogm_to_dict(
            self,
            ogm_instance: AsyncStructuredNode,
            max_depth: int = 10,
            include_properties: bool = True,
            include_relationships: bool = True
    ) -> Dict[str, Any]:
        """Convert OGM node to dictionary."""
        processed: Dict[str, Dict[str, Any]] = {}
        path: Set[str] = set()
        return await self._ogm_to_dict_recursive(
            ogm_instance, processed, path, max_depth,
            include_properties, include_relationships
        )

    async def _ogm_to_dict_recursive(
            self,
            ogm_instance: AsyncStructuredNode,
            processed: Dict[str, Dict[str, Any]],
            path: Set[str],
            max_depth: int,
            include_properties: bool = True,
            include_relationships: bool = True
    ) -> Dict[str, Any]:
        """Recursive implementation of ogm_to_dict with shared prechecks."""
        if not ogm_instance.element_id:
            await self._save_node(ogm_instance)

        ok, element_id, result = self._prepare_ogm_to_dict(
            ogm_instance, processed, path, max_depth, include_properties
        )
        if not ok:
            return result

        if not include_relationships:
            return result

        path.add(element_id)
        try:
            for rel_name, rel_manager in self._iter_ogm_relationship_managers(ogm_instance):
                related_nodes = await self._get_all_related(rel_manager)
                converted = [
                    await self._ogm_to_dict_recursive(
                        node,
                        processed,
                        path,
                        max_depth - 1,
                        include_properties,
                        include_relationships,
                    )
                    for node in related_nodes
                ]
                result[rel_name] = unwrap_relationship_value(rel_manager, rel_name, converted)
        finally:
            path.remove(element_id)

        return result
