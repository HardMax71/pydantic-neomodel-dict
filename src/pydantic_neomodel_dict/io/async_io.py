"""Asynchronous database I/O operations."""

import logging
from contextlib import asynccontextmanager
from typing import List

from neomodel.async_.core import AsyncStructuredNode, adb
from neomodel.exceptions import CardinalityViolation

from ..core.hooks import get_hooks

logger = logging.getLogger(__name__)


async def save_node(node: AsyncStructuredNode) -> None:
    """Save node with hook execution."""
    hooks = get_hooks()

    hooks.execute_before_save(node)
    await node.save()
    hooks.execute_after_save(node)


async def connect_nodes(rel_manager, target: AsyncStructuredNode) -> None:
    """Connect nodes with hook execution. Uses reconnect() for One/ZeroOrOne cardinality."""
    from neomodel.async_.cardinality import One, ZeroOrOne

    hooks = get_hooks()
    source = rel_manager.source
    rel_type = rel_manager.definition.get('relation_type', 'UNKNOWN')

    hooks.execute_before_connect(source, rel_type, target)

    # For One/ZeroOrOne cardinality, use reconnect if connection exists
    if isinstance(rel_manager, (One, ZeroOrOne)):
        count = await rel_manager.count()
        if count:
            # Get the existing node and reconnect to new target
            old_node = await rel_manager.single()
            await rel_manager.reconnect(old_node, target)
        else:
            await rel_manager.connect(target)
    else:
        await rel_manager.connect(target)

    hooks.execute_after_connect(source, rel_type, target)


async def disconnect_nodes(rel_manager, target: AsyncStructuredNode) -> None:
    """Disconnect nodes."""
    await rel_manager.disconnect(target)


async def get_all_related(rel_manager) -> List[AsyncStructuredNode]:
    """Get all related nodes, handling cardinality violations."""
    try:
        return await rel_manager.all()
    except CardinalityViolation:
        return []


async def filter_nodes(ogm_class: type[AsyncStructuredNode], **filters) -> List[AsyncStructuredNode]:
    """Filter nodes by properties."""
    return await ogm_class.nodes.filter(**filters).all()


async def merge_node_on_unique(
    ogm_class: type[AsyncStructuredNode],
    unique_props: dict,
    all_props: dict
) -> AsyncStructuredNode:
    """Atomically merge node matching on unique properties, setting all properties.

    Uses Cypher MERGE to match on unique_props, then SET all_props.
    This is atomic - no race conditions.
    """
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

    results, meta = await adb.cypher_query(query, deflated_non_none)

    if results:
        node_data = results[0][0]
        node = ogm_class.inflate(node_data)
        return node

    raise Exception("MERGE failed to return node")


@asynccontextmanager
async def transaction():
    """Database transaction context."""
    async with adb.transaction:
        yield
