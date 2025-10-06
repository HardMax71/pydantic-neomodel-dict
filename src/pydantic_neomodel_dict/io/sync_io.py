import logging
from contextlib import contextmanager
from typing import List

from neomodel import One, RelationshipManager, StructuredNode, ZeroOrOne, db
from neomodel.exceptions import CardinalityViolation

from ._common import build_merge_query, get_relationship_type, should_use_reconnect
from ..core.hooks import get_hooks

logger = logging.getLogger(__name__)


def save_node(node: StructuredNode) -> None:
    hooks = get_hooks()
    hooks.execute_before_save(node)
    node.save()
    hooks.execute_after_save(node)


def connect_nodes(rel_manager: RelationshipManager, target: StructuredNode) -> None:
    hooks = get_hooks()
    source = rel_manager.source
    rel_type = get_relationship_type(rel_manager)

    hooks.execute_before_connect(source, rel_type, target)

    if should_use_reconnect(rel_manager):
        if len(rel_manager):
            old_node = rel_manager.single()
            rel_manager.reconnect(old_node, target)
        else:
            rel_manager.connect(target)
    else:
        rel_manager.connect(target)

    hooks.execute_after_connect(source, rel_type, target)


def disconnect_nodes(rel_manager: RelationshipManager, target: StructuredNode) -> None:
    rel_manager.disconnect(target)


def get_all_related(rel_manager: RelationshipManager) -> List[StructuredNode]:
    try:
        return list(rel_manager.all())
    except CardinalityViolation:
        return []


def filter_nodes(ogm_class: type[StructuredNode], **filters) -> List[StructuredNode]:
    return list(ogm_class.nodes.filter(**filters))


def merge_node_on_unique(
    ogm_class: type[StructuredNode],
    unique_props: dict,
    all_props: dict
) -> StructuredNode:
    query, params = build_merge_query(ogm_class, unique_props, all_props)
    results, meta = db.cypher_query(query, params)

    if results:
        node_data = results[0][0]
        return ogm_class.inflate(node_data)

    raise Exception("MERGE failed to return node")


@contextmanager
def transaction():
    with db.transaction:
        yield
