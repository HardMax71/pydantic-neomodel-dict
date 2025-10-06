from contextlib import nullcontext
from typing import Any, List, Optional, Set

from pydantic import BaseModel

from pydantic_neomodel_dict.converters._base_converter import BaseConverter


class _DummyConverter(BaseConverter[object, object]):
    def _save_node(self, node: object) -> None:  # pragma: no cover - not used
        pass

    def _get_all_related(self, rel_manager: object) -> list[object]:  # pragma: no cover - not used
        return []

    def _connect_nodes(self, rel_manager: object, target: object) -> None:  # pragma: no cover - not used
        pass

    def _disconnect_nodes(self, rel_manager: object, target: object) -> None:  # pragma: no cover - not used
        pass

    def _merge_node_on_unique(
        self,
        ogm_class: type[object],
        unique_props: dict[str, Any],
        all_props: dict[str, Any]
    ) -> object:  # pragma: no cover - not used
        return object()

    def _transaction(self):  # pragma: no cover - not used
        return nullcontext()


def test_is_list_annotation_various_shapes():
    conv = _DummyConverter()

    assert conv._is_list_annotation(list[int])
    assert conv._is_list_annotation(List[int])
    assert conv._is_list_annotation(Optional[List[int]])
    assert conv._is_list_annotation(list)
    assert not conv._is_list_annotation(Set[str])
    assert not conv._is_list_annotation(int)


class _Child(BaseModel):
    a: int


class _Parent(BaseModel):
    name: str
    age: Optional[int] = None
    child: Optional[_Child] = None
    children: Optional[list[_Child]] = None
    tags: Optional[list[str]] = None


def test_extract_pydantic_properties_skips_nested_models():
    conv = _DummyConverter()

    parent = _Parent(
        name="x",
        age=None,
        child=_Child(a=1),
        children=[_Child(a=2)],
        tags=["t"],
    )

    data = conv._extract_pydantic_properties(parent)
    assert data["name"] == "x"
    assert data["tags"] == ["t"]
    assert "child" not in data
    assert "children" not in data
    assert "age" not in data  # None excluded


def test_filter_defined_properties_and_cache_key():
    class _Prop:
        def __init__(self, unique: bool = False) -> None:
            self.unique_index = unique

    class _OGM:
        @staticmethod
        def defined_properties(*, rels: bool = False, aliases: bool = False, properties: bool = True):
            return {"a": _Prop(False), "u": _Prop(True), "x": _Prop(False)}

    conv = _DummyConverter()
    props = {"a": 1, "u": 2, "x": None, "z": 3}

    filtered, unique = conv._filter_defined_properties(_OGM, props)
    assert filtered == {"a": 1, "u": 2}
    assert unique == {"u": 2}

    key = conv._build_unique_cache_key(_OGM, unique)
    assert isinstance(key, tuple) and len(key) == 2
