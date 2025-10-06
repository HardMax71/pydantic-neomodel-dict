from typing import Any, Protocol, TypeVar


TNode = TypeVar('TNode')


class NodeClassProtocol(Protocol):
    """Protocol for OGM node classes."""

    __name__: str

    @classmethod
    def deflate(cls, properties: dict) -> dict: ...


def get_node_label(ogm_class: type) -> str:
    return getattr(ogm_class, '__label__', ogm_class.__name__)


def build_merge_query(
    ogm_class: type,
    unique_props: dict,
    all_props: dict
) -> tuple[str, dict]:
    deflated_all = ogm_class.deflate(all_props)
    deflated_unique = {
        k: deflated_all[k]
        for k in unique_props.keys()
        if k in deflated_all
    }

    deflated_non_none = {
        k: v
        for k, v in deflated_all.items()
        if v is not None
    }

    label = get_node_label(ogm_class)

    match_parts = [f"{k}: ${k}" for k in deflated_unique.keys()]
    match_clause = "{" + ", ".join(match_parts) + "}"

    set_parts = [f"n.{k} = ${k}" for k in deflated_non_none.keys()]
    set_clause = ", ".join(set_parts)

    query = f"""
    MERGE (n:{label} {match_clause})
    SET {set_clause}
    RETURN n
    """

    return query, deflated_non_none


def should_use_reconnect(rel_manager: Any) -> bool:
    class_name = rel_manager.__class__.__name__
    return class_name in ('One', 'ZeroOrOne', 'AsyncOne', 'AsyncZeroOrOne')


def get_relationship_type(rel_manager: Any) -> str:
    definition = getattr(rel_manager, 'definition', {})
    return definition.get('relation_type', 'UNKNOWN')
