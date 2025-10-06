from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from neomodel.async_.core import AsyncStructuredNode, adb
from neomodel.async_.relationship_manager import AsyncRelationshipManager
from neomodel.exceptions import CardinalityViolation

from ..core.hooks import get_hooks
from ._common import build_merge_query, get_relationship_type, should_use_reconnect


async def save_node(node: AsyncStructuredNode) -> None:
    hooks = get_hooks()
    hooks.execute_before_save(node)
    await node.save()
    hooks.execute_after_save(node)


async def connect_nodes(rel_manager: AsyncRelationshipManager, target: AsyncStructuredNode) -> None:
    hooks = get_hooks()
    source: AsyncStructuredNode = rel_manager.source
    rel_type: str = get_relationship_type(rel_manager)

    hooks.execute_before_connect(source, rel_type, target)

    count: int = await rel_manager.get_len() if should_use_reconnect(rel_manager) else 0
    use_reconnect: bool = count > 0
    old_node = await rel_manager.single() if use_reconnect else None

    if use_reconnect and old_node is not None:
        await rel_manager.reconnect(old_node, target)
    else:
        await rel_manager.connect(target)

    hooks.execute_after_connect(source, rel_type, target)


async def disconnect_nodes(rel_manager: AsyncRelationshipManager, target: AsyncStructuredNode) -> None:
    await rel_manager.disconnect(target)


async def get_all_related(rel_manager: AsyncRelationshipManager) -> list[AsyncStructuredNode]:
    try:
        nodes: list[AsyncStructuredNode] = await rel_manager.all()
        return nodes
    except CardinalityViolation:
        return []

async def merge_node_on_unique(
        ogm_class: type[AsyncStructuredNode],
        unique_props: dict[str, Any],
        all_props: dict[str, Any]
) -> AsyncStructuredNode:
    query: str
    params: dict[str, Any]
    query, params = build_merge_query(ogm_class, unique_props, all_props)
    results: list[list[Any]]
    meta: Any
    results, meta = await adb.cypher_query(query, params)

    if results:
        node_data: Any = results[0][0]
        node: AsyncStructuredNode = ogm_class.inflate(node_data)
        return node

    raise Exception("MERGE failed to return node")


@asynccontextmanager
async def transaction() -> AsyncGenerator[None, None]:
    async with adb.transaction:
        yield
