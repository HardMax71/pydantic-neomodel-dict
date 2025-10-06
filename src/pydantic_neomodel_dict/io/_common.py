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
