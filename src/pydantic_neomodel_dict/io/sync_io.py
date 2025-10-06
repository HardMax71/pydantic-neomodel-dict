"""Synchronous database I/O operations."""

import logging
from contextlib import contextmanager
from typing import List

from neomodel import RelationshipManager, StructuredNode, db
from neomodel.exceptions import CardinalityViolation

from ..core.hooks import get_hooks

logger = logging.getLogger(__name__)


def save_node(node: StructuredNode) -> None:
    """Save node with hook execution."""
    hooks = get_hooks()

    hooks.execute_before_save(node)
    node.save()
    hooks.execute_after_save(node)


def connect_nodes(rel_manager: RelationshipManager, target: StructuredNode) -> None:
    """Connect nodes with hook execution. Uses reconnect() for One/ZeroOrOne cardinality."""
    from neomodel import One, ZeroOrOne

    hooks = get_hooks()
    source = rel_manager.source
    rel_type = rel_manager.definition.get('relation_type', 'UNKNOWN')

    hooks.execute_before_connect(source, rel_type, target)

    # For One/ZeroOrOne cardinality, use reconnect if connection exists
    if isinstance(rel_manager, (One, ZeroOrOne)):
        if len(rel_manager):
            # Get the existing node and reconnect to new target
            old_node = rel_manager.single()
            rel_manager.reconnect(old_node, target)
        else:
            rel_manager.connect(target)
    else:
        rel_manager.connect(target)

    hooks.execute_after_connect(source, rel_type, target)


def disconnect_nodes(rel_manager: RelationshipManager, target: StructuredNode) -> None:
    """Disconnect nodes."""
    rel_manager.disconnect(target)


def get_all_related(rel_manager: RelationshipManager) -> List[StructuredNode]:
    """Get all related nodes, handling cardinality violations."""
    try:
        return list(rel_manager.all())
    except CardinalityViolation:
        return []


def filter_nodes(ogm_class: type[StructuredNode], **filters) -> List[StructuredNode]:
    """Filter nodes by properties."""
    return list(ogm_class.nodes.filter(**filters))


def merge_node_on_unique(
    ogm_class: type[StructuredNode],
    unique_props: dict,
    all_props: dict
) -> StructuredNode:
    """Atomically merge node matching on unique properties, setting all properties.

    Uses Cypher MERGE to match on unique_props, then SET all_props.
    This is atomic - no race conditions.
    """
    from neomodel import db

    deflated_all = ogm_class.deflate(all_props)
    deflated_unique = {k: deflated_all[k] for k in unique_props.keys() if k in deflated_all}

    # Filter out None values - don't overwrite existing data with None
    deflated_non_none = {k: v for k, v in deflated_all.items() if v is not None}

    label = ogm_class.__label__ if hasattr(ogm_class, '__label__') else ogm_class.__name__

    match_parts = [f"{k}: ${k}" for k in deflated_unique.keys()]
    match_clause = "{" + ", ".join(match_parts) + "}"

    # Build SET clause from non-None deflated props only
    set_parts = [f"n.{k} = ${k}" for k in deflated_non_none.keys()]
    set_clause = ", ".join(set_parts)

    query = f"""
    MERGE (n:{label} {match_clause})
    SET {set_clause}
    RETURN n
    """

    results, meta = db.cypher_query(query, deflated_non_none)

    if results:
        node_data = results[0][0]
        node = ogm_class.inflate(node_data)
        return node

    raise Exception("MERGE failed to return node")


@contextmanager
def transaction():
    """Database transaction context."""
    with db.transaction:
        yield
