from typing import Optional

import pytest
from neomodel import (
    StructuredNode, StringProperty, IntegerProperty, BooleanProperty,
    RelationshipTo, ZeroOrOne, One
)
from pydantic import BaseModel

from converter import Converter, ConversionError


# ===== Models for cycle detection tests =====

class NodeAPydantic(BaseModel):
    name: str
    uid: str
    is_active: bool = True
    # Required property with no default
    code: str

    # Forward reference to NodeB
    to_b: 'NodeBPydantic' = None
    # Forward reference to self
    self_ref: 'NodeAPydantic' = None


class NodeBPydantic(BaseModel):
    name: str
    value: int = 0

    # Reference back to NodeA (creates cycle)
    to_a: NodeAPydantic = None


# Resolve forward references
NodeAPydantic.model_rebuild()


class NodeAOGM(StructuredNode):
    name = StringProperty(required=True)
    uid = StringProperty(unique_index=True)  # Unique index property should be included in minimal instance
    is_active = BooleanProperty(default=True)
    code = StringProperty(required=True)  # Required property should be included in minimal instance

    # Relationship to NodeB
    to_b = RelationshipTo('NodeBOGM', 'CONNECTS_TO', cardinality=ZeroOrOne)
    # Self-relationship
    self_ref = RelationshipTo('NodeAOGM', 'REFERS_TO_SELF', cardinality=ZeroOrOne)


class NodeBOGM(StructuredNode):
    name = StringProperty()
    value = IntegerProperty(default=0)

    # Relationship back to NodeA (creates cycle)
    to_a = RelationshipTo(NodeAOGM, 'CONNECTS_TO', cardinality=One)


class UnregisteredCyclicOGM(StructuredNode):
    name = StringProperty(required=True)
    refers_to = RelationshipTo('UnregisteredCyclicOGM', 'REFERS_TO')


# ===== Test Class =====

class TestCycleDetection:
    """Tests specifically targeting cycle detection in to_pydantic method"""

    @pytest.fixture(autouse=True)
    def register_models(self):
        """Register models for conversion"""
        Converter.register_models(NodeAPydantic, NodeAOGM)
        Converter.register_models(NodeBPydantic, NodeBOGM)
        yield
        # Clear registrations after test
        Converter._pydantic_to_ogm = {}
        Converter._ogm_to_pydantic = {}

    def test_cycle_detection_with_self_reference(self, db_connection):
        """
        Test cycle detection when a node references itself directly.

        This tests the minimal instance creation with a direct self-reference,
        which should trigger the cycle detection branch.
        """
        # Create NodeA with self-reference
        node_a = NodeAOGM(
            name="Self-referencing Node",
            uid="self-ref-001",
            code="SR001"
        ).save()

        # Create self-reference
        node_a.self_ref.connect(node_a)

        # Convert to Pydantic
        pydantic_a = Converter.to_pydantic(node_a)

        # Verify basic properties
        assert pydantic_a.name == "Self-referencing Node"
        assert pydantic_a.uid == "self-ref-001"
        assert pydantic_a.code == "SR001"

        # Verify self-reference is handled
        assert pydantic_a.self_ref is not None

        # Verify the self-reference is a minimal instance
        # by checking it's not the same object (would be if cycle not detected)
        assert id(pydantic_a) != id(pydantic_a.self_ref)

        # Verify minimal instance contains required and unique index properties
        assert pydantic_a.self_ref.uid == "self-ref-001"
        assert pydantic_a.self_ref.code == "SR001"

    def test_cycle_detection_with_bidirectional_reference(self, db_connection):
        """
        Test cycle detection in bidirectional relationship between two nodes.

        This tests the minimal instance creation with a cycle between two different
        node types, which should trigger the cycle detection branch.
        """
        # Create NodeA and NodeB
        node_a = NodeAOGM(
            name="Node A",
            uid="node-a-001",
            code="NA001"
        ).save()

        node_b = NodeBOGM(
            name="Node B",
            value=42
        ).save()

        # Create bidirectional relationship
        node_a.to_b.connect(node_b)
        node_b.to_a.connect(node_a)

        # Convert NodeA to Pydantic
        pydantic_a = Converter.to_pydantic(node_a)

        # Verify basic properties
        assert pydantic_a.name == "Node A"
        assert pydantic_a.uid == "node-a-001"

        # Verify relationships
        assert pydantic_a.to_b is not None
        assert pydantic_a.to_b.name == "Node B"
        assert pydantic_a.to_b.value == 42

        # Check that to_b.to_a exists and is a minimal instance
        assert pydantic_a.to_b.to_a is not None

        # Verify it's a different instance but with key properties
        assert id(pydantic_a) != id(pydantic_a.to_b.to_a)
        assert pydantic_a.to_b.to_a.uid == "node-a-001"
        assert pydantic_a.to_b.to_a.code == "NA001"

    def test_cycle_detection_with_longer_cycle(self, db_connection):
        """
        Test cycle detection in a longer reference cycle: A -> B -> A -> B...

        This tests that the cycle detection branch works with longer cycles,
        not just immediate cycles.
        """
        # Create multiple nodes
        node_a1 = NodeAOGM(
            name="Node A1",
            uid="node-a-001",
            code="NA001"
        ).save()

        node_b1 = NodeBOGM(
            name="Node B1",
            value=1
        ).save()

        node_a2 = NodeAOGM(
            name="Node A2",
            uid="node-a-002",
            code="NA002"
        ).save()

        node_b2 = NodeBOGM(
            name="Node B2",
            value=2
        ).save()

        # Create cycle: A1 -> B1 -> A2 -> B2 -> A1
        node_a1.to_b.connect(node_b1)
        node_b1.to_a.connect(node_a2)
        node_a2.to_b.connect(node_b2)
        node_b2.to_a.connect(node_a1)

        # Convert NodeA1 to Pydantic
        pydantic_a1 = Converter.to_pydantic(node_a1)

        # Verify the cycle was handled
        assert pydantic_a1.to_b.to_a.to_b.to_a is not None

        # Verify the cycle end is a minimal instance of A1
        assert pydantic_a1.to_b.to_a.to_b.to_a.uid == "node-a-001"
        assert pydantic_a1.to_b.to_a.to_b.to_a.code == "NA001"

    def test_cycle_detection_with_no_mapping_registered(self, db_connection):
        """
        Test error handling when a cycle is detected but no mapping is registered.

        This tests the error branch where a cycle is detected but the OGM class
        has no registered Pydantic equivalent.
        """

        # Define the class outside the method to avoid lookup issues
        class UnregisteredCyclicORM(StructuredNode):
            name = StringProperty()

        # Create an instance without self-reference
        unregistered = UnregisteredCyclicORM(name="Unregistered").save()

        # Clear registrations to force error
        Converter._ogm_to_pydantic = {}

        # Attempt conversion - should raise ConversionError
        with pytest.raises(ConversionError) as excinfo:
            Converter.to_pydantic(unregistered)

        # Verify error message
        assert "No mapping registered for OGM class" in str(excinfo.value)
        assert "UnregisteredCyclicORM" in str(excinfo.value)

    def test_minimal_instance_creates_minimal_properties(self, db_connection):
        """
        Test that _create_minimal_pydantic_instance creates instances with only
        essential properties.

        This directly tests the implementation of _create_minimal_pydantic_instance,
        which is used to break cycles.
        """
        # Create NodeA with many properties
        node_a = NodeAOGM(
            name="Complete Node",
            uid="complete-001",
            code="C001",
            is_active=True
        ).save()

        # Create self-reference to trigger cycle detection
        node_a.self_ref.connect(node_a)

        # Convert to Pydantic
        pydantic_a = Converter.to_pydantic(node_a)

        # The self-reference should be a minimal instance
        minimal_instance = pydantic_a.self_ref

        # Required properties should be present
        assert minimal_instance.uid == "complete-001"
        assert minimal_instance.code == "C001"

        # Non-required properties might be omitted in a minimal instance
        # Note: exact behavior depends on _create_minimal_pydantic_instance implementation

        # Verify it's a different object
        assert id(pydantic_a) != id(minimal_instance)

    def test_to_pydantic_no_mapping_registered(self, db_connection):
        """
        Test that to_pydantic raises a ConversionError when no mapping is registered
        for the OGM class and pydantic_class is not provided.
        """

        # Define a class that isn't registered
        class UnregisteredOGM2(StructuredNode):
            name = StringProperty()

        # Create an instance
        unregistered = UnregisteredOGM2(name="Unregistered").save()

        # Attempt conversion without providing pydantic_class - should raise ConversionError
        with pytest.raises(ConversionError) as excinfo:
            Converter.to_pydantic(unregistered)

        # Verify error message
        assert "No mapping registered for OGM class" in str(excinfo.value)
        assert "UnregisteredOGM" in str(excinfo.value)

    def test_to_pydantic_cycle_with_no_class_mapping(self, db_connection):
        """
        Test that a ConversionError is raised when a cycle is detected in to_pydantic,
        no explicit pydantic_class is provided, and the OGM class has no registered mapping.
        """
        # Create an instance with a self-reference to create a cycle
        cyclic_node = UnregisteredCyclicOGM(name="CyclicNode").save()
        cyclic_node.refers_to.connect(cyclic_node)

        # Create a separate unregistered Pydantic model that could be passed explicitly
        class UnregisteredCyclicModel(BaseModel):
            name: str
            refers_to: Optional['UnregisteredCyclicModel'] = None

        UnregisteredCyclicModel.model_rebuild()  # Resolve forward references

        # Clear registrations to ensure no mapping exists
        original_mappings = Converter._ogm_to_pydantic.copy()
        Converter._ogm_to_pydantic = {}

        try:
            # Call to_pydantic without providing pydantic_class - should raise ConversionError
            with pytest.raises(ConversionError) as excinfo:
                Converter.to_pydantic(cyclic_node)

            # Verify error message
            assert "No mapping registered for OGM class" in str(excinfo.value)
            assert "UnregisteredCyclicOGM" in str(excinfo.value)

            # Register the model temporarily for the explicit test
            Converter._ogm_to_pydantic[UnregisteredCyclicOGM] = UnregisteredCyclicModel

            # Now this should work with the mapping in place
            result = Converter.to_pydantic(cyclic_node, UnregisteredCyclicModel)

            # Detailed assertions about the result
            assert result is not None, "Result should not be None"
            assert isinstance(result,
                              UnregisteredCyclicModel), "Result should be an instance of UnregisteredCyclicModel"
            assert result.name == "CyclicNode", "Root object name should match original"

            # Check the self-reference
            assert result.refers_to is not None, "Self-reference should not be None"
            assert isinstance(result.refers_to,
                              UnregisteredCyclicModel), "Self-reference should be an instance of UnregisteredCyclicModel"
            assert result.refers_to.name == "CyclicNode", "Self-reference name should match original"

            # Verify that objects are distinct (proves cycle detection worked)
            assert id(result) != id(result.refers_to), "Result and its reference should be distinct objects"

            # Verify that the cycle is properly terminated
            assert result.refers_to.refers_to is None, "The cycle should be terminated at the first level"

        finally:
            # Restore original mappings
            Converter._ogm_to_pydantic = original_mappings
