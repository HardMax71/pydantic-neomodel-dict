import pytest
from neomodel import (
    BooleanProperty,
    IntegerProperty,
    One,
    RelationshipTo,
    StringProperty,
    StructuredNode,
    ZeroOrOne,
)
from pydantic import BaseModel

from pydantic_neomodel_dict import Converter

# ===== Models for cycle detection tests =====

class CycleAPydantic(BaseModel):
    name: str
    uid: str
    priority: int = 0
    is_active: bool = True

    # Forward reference to self
    parent: 'CycleAPydantic' = None
    # Forward reference to child nodes
    children: list['CycleAPydantic'] = []


# Resolve forward references
CycleAPydantic.model_rebuild()


class CycleAOGM(StructuredNode):
    name = StringProperty(required=True)
    uid = StringProperty(unique_index=True)
    priority = IntegerProperty(default=0)
    is_active = BooleanProperty(default=True)

    # Relationship to self as parent
    parent = RelationshipTo('CycleAOGM', 'HAS_PARENT', cardinality=ZeroOrOne)
    # Relationship to children
    children = RelationshipTo('CycleAOGM', 'HAS_CHILDREN')


# ===== Models for bidirectional cycle tests =====

class BidirAPydantic(BaseModel):
    name: str
    code: str

    # Forward reference to B
    to_b: 'BidirBPydantic' = None


class BidirBPydantic(BaseModel):
    name: str
    value: int = 0

    # Reference back to A
    to_a: BidirAPydantic = None


# Resolve forward references
BidirAPydantic.model_rebuild()


class BidirAOGM(StructuredNode):
    name = StringProperty(required=True)
    code = StringProperty(unique_index=True)

    # Relationship to B
    to_b = RelationshipTo('BidirBOGM', 'CONNECTS_TO', cardinality=ZeroOrOne)


class BidirBOGM(StructuredNode):
    name = StringProperty()
    value = IntegerProperty(default=0)

    # Relationship back to A
    to_a = RelationshipTo(BidirAOGM, 'CONNECTS_TO', cardinality=One)


# ===== Models for deep cycle tests =====

class DeepNodePydantic(BaseModel):
    name: str
    level: int  # This is a required field
    next_node: 'DeepNodePydantic' = None


# Resolve forward references
DeepNodePydantic.model_rebuild()


class DeepNodeOGM(StructuredNode):
    name = StringProperty()
    # Make `level` required so it gets included in minimal instances
    level = IntegerProperty(required=True)

    # Relationship to next node
    next_node = RelationshipTo('DeepNodeOGM', 'NEXT', cardinality=ZeroOrOne)


# ===== Test Class =====

class TestCycleDetection:
    """Tests specifically targeting cycle detection in to_pydantic method"""

    @pytest.fixture(autouse=True)
    def register_models(self):
        """Register models for conversion"""
        Converter.register_models(CycleAPydantic, CycleAOGM)
        Converter.register_models(BidirAPydantic, BidirAOGM)
        Converter.register_models(BidirBPydantic, BidirBOGM)
        Converter.register_models(DeepNodePydantic, DeepNodeOGM)
        yield
        # Clear registrations after test
        Converter._pydantic_to_ogm = {}
        Converter._ogm_to_pydantic = {}

    def test_self_reference_creates_distinct_objects(self, db_connection):
        """
        Test that a self-reference creates distinct Python objects.

        This specifically tests the cycle detection branch with direct self-reference.
        """
        # Create a node with self-reference
        node = CycleAOGM(
            name="Self-Referencing Node",
            uid="SR001",
            priority=1
        ).save()

        # Create self-reference
        node.parent.connect(node)

        # Convert to Pydantic
        pydantic_node = Converter.to_pydantic(node)

        # Verify basic properties
        assert pydantic_node.name == "Self-Referencing Node"
        assert pydantic_node.uid == "SR001"
        assert pydantic_node.priority == 1

        # Verify self-reference exists and has the right properties
        assert pydantic_node.parent is not None
        assert pydantic_node.parent.uid == "SR001"

        # The key test: verify objects are distinct Python instances
        assert pydantic_node is not pydantic_node.parent

        # Verify we can access further properties without recursion issues
        assert pydantic_node.parent.name == "Self-Referencing Node"

    def test_bidirectional_reference_creates_distinct_objects(self, db_connection):
        """
        Test that a bidirectional reference cycle creates distinct Python objects.

        This tests the cycle detection branch with bidirectional references.
        """
        # Create nodes A and B
        node_a = BidirAOGM(name="Node A", code="A001").save()
        node_b = BidirBOGM(name="Node B", value=42).save()

        # Create bidirectional relationship
        node_a.to_b.connect(node_b)
        node_b.to_a.connect(node_a)

        # Convert to Pydantic
        pydantic_a = Converter.to_pydantic(node_a)

        # Verify basic properties
        assert pydantic_a.name == "Node A"
        assert pydantic_a.code == "A001"

        # Verify relationships exist
        assert pydantic_a.to_b is not None
        assert pydantic_a.to_b.name == "Node B"
        assert pydantic_a.to_b.value == 42

        # Verify the cycle is handled with a distinct object
        assert pydantic_a.to_b.to_a is not None
        assert pydantic_a.to_b.to_a.code == "A001"
        assert pydantic_a is not pydantic_a.to_b.to_a

    def test_multiple_distinct_instances_for_same_node(self, db_connection):
        """
        Test that multiple references to the same node create distinct Python objects.

        This tests that the cycle detection logic creates separate instances each time.
        """
        # Create a tree-like structure with parent and children
        parent = CycleAOGM(name="Parent", uid="P001", priority=10).save()
        child1 = CycleAOGM(name="Child 1", uid="C001", priority=5).save()
        child2 = CycleAOGM(name="Child 2", uid="C002", priority=5).save()

        # Create relationships with parent
        child1.parent.connect(parent)
        child2.parent.connect(parent)

        # Also connect parent to children (creates inverse relationships)
        parent.children.connect(child1)
        parent.children.connect(child2)

        # Convert parent to Pydantic
        pydantic_parent = Converter.to_pydantic(parent)

        # Verify children exist
        assert len(pydantic_parent.children) == 2

        # Get references to the same parent via different paths
        path1_parent = pydantic_parent
        path2_parent = pydantic_parent.children[0].parent
        path3_parent = pydantic_parent.children[1].parent

        # Verify they are all the same underlying data
        assert path1_parent.uid == "P001"
        assert path2_parent.uid == "P001"
        assert path3_parent.uid == "P001"

        # But distinct Python objects
        assert path1_parent is not path2_parent
        assert path1_parent is not path3_parent
        assert path2_parent is not path3_parent  # Even different cycle paths get distinct objects

    def test_deep_cycle_references(self, db_connection):
        """
        Test handling of deeper cycles with multiple levels.

        This tests cycle detection in longer chains where the cycle appears after
        several links.
        """
        # Create a chain of nodes that eventually cycles back
        node1 = DeepNodeOGM(name="Level 1", level=1).save()
        node2 = DeepNodeOGM(name="Level 2", level=2).save()
        node3 = DeepNodeOGM(name="Level 3", level=3).save()
        node4 = DeepNodeOGM(name="Level 4", level=4).save()
        node5 = DeepNodeOGM(name="Level 5", level=5).save()

        # Link them in a chain: 1 -> 2 -> 3 -> 4 -> 5 -> 3 (cycle back to 3)
        node1.next_node.connect(node2)
        node2.next_node.connect(node3)
        node3.next_node.connect(node4)
        node4.next_node.connect(node5)
        node5.next_node.connect(node3)  # Creates cycle

        # Convert to Pydantic starting from node1
        pydantic_node1 = Converter.to_pydantic(node1)

        # Traverse the chain
        pydantic_node2 = pydantic_node1.next_node
        pydantic_node3 = pydantic_node2.next_node
        pydantic_node4 = pydantic_node3.next_node
        pydantic_node5 = pydantic_node4.next_node
        pydantic_node3_cycle = pydantic_node5.next_node

        # Verify chain properties
        assert pydantic_node1.level == 1
        assert pydantic_node2.level == 2
        assert pydantic_node3.level == 3
        assert pydantic_node4.level == 4
        assert pydantic_node5.level == 5

        # Verify cycle is detected and handled
        assert pydantic_node3_cycle is not None
        assert pydantic_node3_cycle.level == 3
        assert pydantic_node3_cycle.name == "Level 3"

        # Verify distinct object creation in cycle
        assert pydantic_node3 is not pydantic_node3_cycle

    def test_minimal_instance_has_essential_properties(self, db_connection):
        """
        Test that minimal instances created for cycles have essential properties.

        This tests that _create_minimal_pydantic_instance correctly prioritizes
        required and unique index properties.
        """
        # Create a node with self-reference
        node = CycleAOGM(
            name="Complete Node",
            uid="C001",
            priority=99,
            is_active=False
        ).save()

        # Create self-reference
        node.parent.connect(node)

        # Convert to Pydantic
        pydantic_node = Converter.to_pydantic(node)

        # Get minimal instance created for cycle
        minimal_instance = pydantic_node.parent

        # Verify it contains the essential properties
        assert minimal_instance.name == "Complete Node"  # required property
        assert minimal_instance.uid == "C001"  # unique index property

        # Optional properties might or might not be included based on implementation
        # The minimal instance should at least have the key properties

        # Verify it's a different object
        assert pydantic_node is not minimal_instance

    def test_minimal_instance_not_stored_in_processed_objects(self, db_connection):
        """
        Test that minimal instances aren't stored in processed_objects.

        This test verifies that we create a new minimal instance each time we encounter
        the same node in a cycle, ensuring distinct Python objects.
        """
        # Create nodes with relationships that will create cycles
        node1 = CycleAOGM(name="Node 1", uid="N1").save()
        node2 = CycleAOGM(name="Node 2", uid="N2").save()

        # Create bidirectional relationship
        node1.children.connect(node2)
        node2.parent.connect(node1)

        # Also create self-reference on node1
        node1.parent.connect(node1)

        # Convert to Pydantic
        pydantic_node1 = Converter.to_pydantic(node1)

        # There should be three distinct references to node1:
        # 1. The original pydantic_node1
        # 2. The self-reference via parent
        # 3. The cycle reference via children -> parent

        ref1 = pydantic_node1
        ref2 = pydantic_node1.parent  # self-reference
        ref3 = pydantic_node1.children[0].parent  # cycle

        # All references should point to the same underlying data
        assert ref1.uid == "N1"
        assert ref2.uid == "N1"
        assert ref3.uid == "N1"

        # But should be distinct Python objects
        assert ref1 is not ref2
        assert ref1 is not ref3
        assert ref2 is not ref3  # Even different cycle paths get distinct objects
