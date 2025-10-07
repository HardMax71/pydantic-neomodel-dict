from typing import Any, Dict, List, Optional, Set

from neomodel import RelationshipManager, StructuredNode
from pydantic import BaseModel

from ..io import sync_io
from ..io._common import unwrap_relationship_value
from ._base_converter import BaseConverter


class SyncConverter(BaseConverter[StructuredNode, RelationshipManager]):
    """Synchronous converter between Pydantic models, OGM nodes, and dictionaries."""

    def _save_node(self, node: StructuredNode) -> None:
        sync_io.save_node(node)

    def _get_all_related(self, rel_manager: RelationshipManager) -> List[StructuredNode]:
        return sync_io.get_all_related(rel_manager)

    def _connect_nodes(self, rel_manager: RelationshipManager, target: StructuredNode) -> None:
        sync_io.connect_nodes(rel_manager, target)

    def _disconnect_nodes(self, rel_manager: RelationshipManager, target: StructuredNode) -> None:
        sync_io.disconnect_nodes(rel_manager, target)

    def _merge_node_on_unique(
            self,
            ogm_class: type[StructuredNode],
            unique_props: Dict[str, Any],
            all_props: Dict[str, Any]
    ) -> StructuredNode:
        return sync_io.merge_node_on_unique(ogm_class, unique_props, all_props)

    def _transaction(self):  # type: ignore[no-untyped-def]
        return sync_io.transaction()

    def to_ogm(
            self,
            pydantic_instance: BaseModel,
            ogm_class: Optional[type[StructuredNode]] = None,
            max_depth: int = 10
    ) -> StructuredNode:
        """Convert a Pydantic model to an OGM node."""
        with self._transaction():
            processed: Dict[int, StructuredNode] = {}
            return self._to_ogm_recursive(
                pydantic_instance,
                ogm_class,
                processed,
                max_depth
            )

    def batch_to_ogm(
            self,
            pydantic_instances: List[BaseModel],
            ogm_class: Optional[type[StructuredNode]] = None,
            max_depth: int = 10
    ) -> List[StructuredNode]:
        """Batch convert Pydantic models to OGM nodes in one transaction."""
        with self._transaction():
            processed: Dict[int, StructuredNode] = {}
            return [
                self._to_ogm_recursive(instance, ogm_class, processed, max_depth)
                for instance in pydantic_instances
            ]

    def batch_to_pydantic(
            self,
            ogm_instances: List[StructuredNode],
            pydantic_class: Optional[type[BaseModel]] = None,
            max_depth: int = 10
    ) -> List[BaseModel]:
        """Batch convert OGM nodes to Pydantic models."""
        return [self.to_pydantic(instance, pydantic_class, max_depth) for instance in ogm_instances]

    def batch_dict_to_ogm(
            self,
            data_dicts: List[Dict[str, Any]],
            ogm_class: type[StructuredNode],
            max_depth: int = 10
    ) -> List[StructuredNode]:
        """Batch convert dictionaries to OGM nodes in one transaction."""
        with self._transaction():
            processed: Dict[int, StructuredNode] = {}
            return [
                self._dict_to_ogm_recursive(data, ogm_class, processed, max_depth)
                for data in data_dicts
            ]

    def batch_ogm_to_dict(
            self,
            ogm_instances: List[StructuredNode],
            max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """Batch convert OGM nodes to dictionaries."""
        return [self.ogm_to_dict(instance, max_depth) for instance in ogm_instances]

    def _to_ogm_recursive(
            self,
            pydantic_instance: BaseModel,
            ogm_class: Optional[type[StructuredNode]],
            processed: Dict,  # Mixed keys: int for circular refs, tuple for deduplication
            max_depth: int
    ) -> StructuredNode:
        """Recursive implementation of to_ogm using shared prep/finalize helpers."""
        existing, resolved_ogm, obj_id, filtered_props, unique_props, cache_key = self._to_ogm_prepare(
            pydantic_instance, ogm_class, processed, max_depth
        )

        if existing is not None:
            return existing

        ogm_instance = self._get_or_create_node(filtered_props, resolved_ogm)
        self._to_ogm_finalize(processed, obj_id, ogm_instance, unique_props, cache_key)

        if max_depth > 0:
            self._sync_relationships_from_pydantic(
                pydantic_instance,
                ogm_instance,
                processed,
                max_depth - 1
            )

        return ogm_instance

    def _get_or_create_node(
            self,
            properties: Dict[str, Any],
            ogm_class: type[StructuredNode]
    ) -> StructuredNode:
        """Get or create node, with atomic upsert for unique properties."""
        filtered_props, unique_props = self._filter_defined_properties(ogm_class, properties)

        if unique_props:
            return self._merge_node_on_unique(ogm_class, unique_props, filtered_props)

        node = ogm_class(**filtered_props)
        self._save_node(node)
        return node

    def _sync_relationships_from_pydantic(
            self,
            pydantic_instance: BaseModel,
            ogm_instance: StructuredNode,
            processed: Dict[int, StructuredNode],
            max_depth: int
    ) -> None:
        """Synchronize relationships from Pydantic to OGM."""
        for rel_manager, target_ogm_class, items in self._enumerate_pydantic_relationships(
            pydantic_instance, ogm_instance
        ):
            related_nodes: List[StructuredNode] = [
                self._to_ogm_recursive(item, target_ogm_class, processed, max_depth)
                for item in items
            ]
            self._sync_relationship(rel_manager, related_nodes)

    def _sync_relationship(
            self,
            rel_manager: RelationshipManager,
            target_nodes: List[StructuredNode]
    ) -> None:
        """Synchronize relationship to match target nodes exactly.

        Adds missing connections, removes extra connections.
        """
        existing = self._get_all_related(rel_manager)
        to_add, to_remove, targets_by_id, existing_by_id = self._relationship_delta(existing, target_nodes)

        for node_id in to_add:
            self._connect_nodes(rel_manager, targets_by_id[node_id])

        for node_id in to_remove:
            self._disconnect_nodes(rel_manager, existing_by_id[node_id])

    def to_pydantic(
            self,
            ogm_instance: StructuredNode,
            pydantic_class: Optional[type[BaseModel]] = None,
            max_depth: int = 10
    ) -> BaseModel:
        """Convert an OGM node to a Pydantic model."""
        processed: Dict[str, BaseModel] = {}
        path: Set[str] = set()
        return self._to_pydantic_recursive(
            ogm_instance,
            pydantic_class,
            processed,
            path,
            max_depth
        )

    def _to_pydantic_recursive(
            self,
            ogm_instance: StructuredNode,
            pydantic_class: Optional[type[BaseModel]],
            processed: Dict[str, BaseModel],
            path: Set[str],
            max_depth: int
    ) -> BaseModel:
        """Recursive implementation of to_pydantic."""
        if not ogm_instance.element_id:
            self._save_node(ogm_instance)

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
            self._load_pydantic_relationships(
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

    def _load_pydantic_relationships(
            self,
            ogm_instance: StructuredNode,
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
            related_nodes = self._get_all_related(rel_manager)
            converted = [
                self._to_pydantic_recursive(
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

    def dict_to_ogm(
            self,
            data: Dict[str, Any],
            ogm_class: type[StructuredNode],
            max_depth: int = 10
    ) -> StructuredNode:
        """Convert dictionary to OGM node."""
        with self._transaction():
            processed: Dict[int, StructuredNode] = {}
            return self._dict_to_ogm_recursive(data, ogm_class, processed, max_depth)

    def _dict_to_ogm_recursive(
            self,
            data: Dict[str, Any],
            ogm_class: type[StructuredNode],
            processed: Dict,  # Mixed keys: int for circular refs, tuple for deduplication
            max_depth: int
    ) -> StructuredNode:
        """Recursive implementation of dict_to_ogm using shared preparation."""
        existing, data_id, properties, relationships, cache_key, defined_rels = (
            self._dict_to_ogm_prepare(data, ogm_class, processed, max_depth)
        )

        if existing is not None:
            return existing

        node = self._get_or_create_node(properties, ogm_class)
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
                related_nodes: List[StructuredNode] = [
                    self._dict_to_ogm_recursive(item, target_class, processed, max_depth - 1)
                    for item in items
                ]

                self._sync_relationship(rel_manager, related_nodes)

        return node

    def ogm_to_dict(
            self,
            ogm_instance: StructuredNode,
            max_depth: int = 10,
            include_properties: bool = True,
            include_relationships: bool = True
    ) -> Dict[str, Any]:
        """Convert OGM node to dictionary."""
        processed: Dict[str, Dict[str, Any]] = {}
        path: Set[str] = set()
        return self._ogm_to_dict_recursive(
            ogm_instance, processed, path, max_depth,
            include_properties, include_relationships
        )

    def _ogm_to_dict_recursive(
            self,
            ogm_instance: StructuredNode,
            processed: Dict[str, Dict[str, Any]],
            path: Set[str],
            max_depth: int,
            include_properties: bool = True,
            include_relationships: bool = True
    ) -> Dict[str, Any]:
        """Recursive implementation of ogm_to_dict."""
        if not ogm_instance.element_id:
            self._save_node(ogm_instance)

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
                related_nodes = self._get_all_related(rel_manager)
                converted = [
                    self._ogm_to_dict_recursive(
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
