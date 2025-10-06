import logging
from contextlib import asynccontextmanager
from typing import List

from neomodel.async_.cardinality import AsyncOne, AsyncZeroOrOne
from neomodel.async_.core import AsyncStructuredNode, adb
from neomodel.exceptions import CardinalityViolation

from ._common import build_merge_query, get_relationship_type, should_use_reconnect
from ..core.hooks import get_hooks

logger = logging.getLogger(__name__)


async def save_node(node: AsyncStructuredNode) -> None:
    hooks = get_hooks()
    hooks.execute_before_save(node)
    await node.save()
    hooks.execute_after_save(node)


async def connect_nodes(rel_manager, target: AsyncStructuredNode) -> None:
    hooks = get_hooks()
    source = rel_manager.source
    rel_type = get_relationship_type(rel_manager)

    hooks.execute_before_connect(source, rel_type, target)

    if should_use_reconnect(rel_manager):
        count = await rel_manager.count()
        if count:
            old_node = await rel_manager.single()
            await rel_manager.reconnect(old_node, target)
        else:
            await rel_manager.connect(target)
    else:
        await rel_manager.connect(target)

    hooks.execute_after_connect(source, rel_type, target)


async def disconnect_nodes(rel_manager, target: AsyncStructuredNode) -> None:
    await rel_manager.disconnect(target)


async def get_all_related(rel_manager) -> List[AsyncStructuredNode]:
    try:
        return await rel_manager.all()
    except CardinalityViolation:
        return []


async def filter_nodes(ogm_class: type[AsyncStructuredNode], **filters) -> List[AsyncStructuredNode]:
    return await ogm_class.nodes.filter(**filters).all()


async def merge_node_on_unique(
    ogm_class: type[AsyncStructuredNode],
    unique_props: dict,
    all_props: dict
) -> AsyncStructuredNode:
    query, params = build_merge_query(ogm_class, unique_props, all_props)
    results, meta = await adb.cypher_query(query, params)

    if results:
        node_data = results[0][0]
        return ogm_class.inflate(node_data)

    raise Exception("MERGE failed to return node")


@asynccontextmanager
async def transaction():
    async with adb.transaction:
        yield
