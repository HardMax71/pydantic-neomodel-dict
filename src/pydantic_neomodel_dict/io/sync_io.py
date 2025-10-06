from contextlib import contextmanager
from typing import Any, Generator

from neomodel import RelationshipManager, StructuredNode, db
from neomodel.exceptions import CardinalityViolation

from ..core.hooks import get_hooks
from ._common import build_merge_query, get_relationship_type, should_use_reconnect


def save_node(node: StructuredNode) -> None:
    hooks = get_hooks()
    hooks.execute_before_save(node)
    node.save()
    hooks.execute_after_save(node)


def connect_nodes(rel_manager: RelationshipManager, target: StructuredNode) -> None:
    hooks = get_hooks()
    source: StructuredNode = rel_manager.source
    rel_type: str = get_relationship_type(rel_manager)

    hooks.execute_before_connect(source, rel_type, target)

    use_reconnect: bool = should_use_reconnect(rel_manager) and len(rel_manager) > 0
    old_node = rel_manager.single() if use_reconnect else None

    if use_reconnect and old_node is not None:
        rel_manager.reconnect(old_node, target)
    else:
        rel_manager.connect(target)

    hooks.execute_after_connect(source, rel_type, target)


def disconnect_nodes(rel_manager: RelationshipManager, target: StructuredNode) -> None:
    rel_manager.disconnect(target)


def get_all_related(rel_manager: RelationshipManager) -> list[StructuredNode]:
    try:
        return list(rel_manager.all())
    except CardinalityViolation:
        return []

def merge_node_on_unique(
        ogm_class: type[StructuredNode],
        unique_props: dict[str, Any],
        all_props: dict[str, Any]
) -> StructuredNode:
    query: str
    params: dict[str, Any]
    query, params = build_merge_query(ogm_class, unique_props, all_props)
    results: list[list[Any]]
    meta: Any
    results, meta = db.cypher_query(query, params)

    if results:
        node_data: Any = results[0][0]
        node: StructuredNode = ogm_class.inflate(node_data)
        return node

    raise Exception("MERGE failed to return node")


@contextmanager
def transaction() -> Generator[None, None, None]:
    with db.transaction:
        yield
