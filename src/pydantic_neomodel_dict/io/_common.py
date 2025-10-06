from typing import Any, Union

from neomodel import RelationshipManager, StructuredNode
from neomodel.async_.core import AsyncStructuredNode
from neomodel.async_.relationship_manager import AsyncRelationshipManager


def get_node_label(ogm_class: type[Union[StructuredNode, AsyncStructuredNode]]) -> str:
    return str(getattr(ogm_class, '__label__', ogm_class.__name__))


def build_merge_query(
    ogm_class: type[Union[StructuredNode, AsyncStructuredNode]],
    unique_props: dict[str, Any],
    all_props: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    deflated_all: dict[str, Any] = ogm_class.deflate(all_props)
    deflated_unique: dict[str, Any] = {
        k: deflated_all[k]
        for k in unique_props.keys()
        if k in deflated_all
    }

    deflated_non_none: dict[str, Any] = {
        k: v
        for k, v in deflated_all.items()
        if v is not None
    }

    label: str = get_node_label(ogm_class)

    match_parts: list[str] = [f"{k}: ${k}" for k in deflated_unique.keys()]
    match_clause: str = "{" + ", ".join(match_parts) + "}"

    set_parts: list[str] = [f"n.{k} = ${k}" for k in deflated_non_none.keys()]
    set_clause: str = ", ".join(set_parts)

    query: str = f"""
    MERGE (n:{label} {match_clause})
    SET {set_clause}
    RETURN n
    """

    return query, deflated_non_none


def should_use_reconnect(rel_manager: Union[RelationshipManager, AsyncRelationshipManager]) -> bool:
    class_name: str = rel_manager.__class__.__name__
    return class_name in ('One', 'ZeroOrOne', 'AsyncOne', 'AsyncZeroOrOne')


def get_relationship_type(rel_manager: Union[RelationshipManager, AsyncRelationshipManager]) -> str:
    definition: dict[str, Any] = getattr(rel_manager, 'definition', {})
    return str(definition.get('relation_type', 'UNKNOWN'))


def is_single_cardinality(rel_manager: Union[RelationshipManager, AsyncRelationshipManager]) -> bool:
    """Return True if relationship is single-cardinality (One/ZeroOrOne variants).

    Mirrors the check used for reconnect logic, exposed with a clearer name for
    converters to decide when to unwrap single items in dict output.
    """
    return should_use_reconnect(rel_manager)


def unwrap_relationship_value(
    rel_manager: Union[RelationshipManager, AsyncRelationshipManager],
    rel_name: str,
    converted: list[dict]
) -> Any:
    """Determine the appropriate representation for a relationship's converted list.

    - Unwraps to single dict for single-cardinality, incoming, or self-ref chains
    - Returns None for empty single-cardinality
    - Otherwise returns the list unchanged
    """
    single = is_single_cardinality(rel_manager)
    rel_definition: dict[str, Any] = getattr(rel_manager, 'definition', {})
    is_incoming: bool = rel_definition.get('direction') == -1  # RelationshipFrom

    # Check if self-referencing (same source and target class)
    target_class = rel_definition.get('node_class')
    source_class = rel_manager.source.__class__
    is_self_referencing = (
        target_class == source_class or target_class == getattr(source_class, '__name__', None)
    )

    # For self-referencing: check if ALL nodes in this relationship have exactly 1 link
    # This indicates a linked-list/chain structure that should be unwrapped
    unwrap_self_ref = False
    if is_self_referencing and len(converted) == 1:
        target_node = converted[0]
        if isinstance(target_node, dict):
            target_rel_value = target_node.get(rel_name)
            # Unwrap if target also has single item (or is terminal/cycle)
            if target_rel_value is None or isinstance(target_rel_value, dict):
                unwrap_self_ref = True

    should_unwrap_single = single or is_incoming or unwrap_self_ref

    if len(converted) == 1 and should_unwrap_single:
        return converted[0]
    if len(converted) == 0 and single:
        return None
    return converted
