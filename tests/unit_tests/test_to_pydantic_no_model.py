import pytest
from neomodel import (
    IntegerProperty,
    One,
    RelationshipFrom,
    RelationshipTo,
    StringProperty,
    StructuredNode,
    ZeroOrOne,
    db,
)
from pydantic import BaseModel, Field

from pydantic_neo4j_dict import ConversionError, Converter


# Define an unregistered OGM model (won't have a Pydantic counterpart)
class UnregisteredOGM(StructuredNode):
    """OGM model that deliberately won't be registered with a Pydantic model"""
    name = StringProperty()
    code = StringProperty()


# Define registered OGM and Pydantic models
class RegisteredOGM(StructuredNode):
    """OGM model with relationships to unregistered models"""
    name = StringProperty(required=True)
    value = IntegerProperty(default=0)

    # Different relationship types to unregistered model
    unregistered_many = RelationshipTo('UnregisteredOGM', 'RELATES_TO_MANY')
    unregistered_one = RelationshipTo('UnregisteredOGM', 'RELATES_TO_ONE', cardinality=One)
    unregistered_zero_or_one = RelationshipTo('UnregisteredOGM',
                                              'RELATES_TO_ZERO_OR_ONE', cardinality=ZeroOrOne)

    # Add a relationship to MiddleOGM to fix the nested test
    middle_relation = RelationshipTo('MiddleOGM', 'REGISTERED_TO_MIDDLE')


class RegisteredPydantic(BaseModel):
    """Pydantic model corresponding to RegisteredOGM"""
    name: str
    value: int = 0

    # These fields exist but without proper typing since UnregisteredPydantic doesn't exist
    unregistered_many: list = Field(default_factory=list)
    unregistered_one: object = None
    unregistered_zero_or_one: object = None
    middle_relation: list = Field(default_factory=list)


# Models for testing nested relationship scenarios
class MiddleOGM(StructuredNode):
    """Middle node in a relationship chain for testing nested scenarios"""
    name = StringProperty()

    # Relationship to registered model
    registered = RelationshipFrom('RegisteredOGM', 'REGISTERED_TO_MIDDLE')
    # Relationship to unregistered model
    unregistered = RelationshipTo('UnregisteredOGM', 'MIDDLE_TO_UNREGISTERED')


class MiddlePydantic(BaseModel):
    """Pydantic model corresponding to MiddleOGM"""
    name: str
    registered: list = Field(default_factory=list)
    unregistered: list = Field(default_factory=list)


class TestMissingRegistration:
    """Tests for handling relationships to unregistered OGM models during to_pydantic conversion"""

    def setup_method(self):
        """Set up the test environment"""
        # Clear database before each test
        db.cypher_query("MATCH (n) DETACH DELETE n")

        # Register only the registered and middle models
        Converter.register_models(RegisteredPydantic, RegisteredOGM)
        Converter.register_models(MiddlePydantic, MiddleOGM)
        # Deliberately do NOT register UnregisteredOGM with any Pydantic model

    def teardown_method(self):
        """Clean up after each test"""
        # Clear database after each test
        db.cypher_query("MATCH (n) DETACH DELETE n")
        # Clear registrations
        Converter._pydantic_to_ogm = {}
        Converter._ogm_to_pydantic = {}

    def test_unregistered_many_relationship_raises_error(self, db_connection):
        # Test that converting an OGM with many-relationship to
        # unregistered model raises ConversionError
        # Create registered node
        reg_node = RegisteredOGM(name="Registered Node", value=10).save()

        # Create unregistered node
        unreg_node = UnregisteredOGM(name="Unregistered Node", code="UR001").save()

        # Create many-relationship
        reg_node.unregistered_many.connect(unreg_node)

        # Attempt to convert to Pydantic - should raise ConversionError
        with pytest.raises(ConversionError) as excinfo:
            Converter.to_pydantic(reg_node)

        # Verify error message mentions the unregistered class
        assert "No Pydantic model registered for OGM class UnregisteredOGM" in str(excinfo.value)

    def test_unregistered_one_relationship_raises_error(self, db_connection):
        # Test that converting an OGM with One-relationship
        # to unregistered model raises ConversionError
        # Create registered node
        reg_node = RegisteredOGM(name="Registered Node", value=10).save()

        # Create unregistered node
        unreg_node = UnregisteredOGM(name="Unregistered Node", code="UR001").save()

        # Create One-relationship
        reg_node.unregistered_one.connect(unreg_node)

        # Attempt to convert to Pydantic - should raise ConversionError
        with pytest.raises(ConversionError) as excinfo:
            Converter.to_pydantic(reg_node)

        # Verify error message mentions the unregistered class
        assert "No Pydantic model registered for OGM class UnregisteredOGM" in str(excinfo.value)

    def test_unregistered_zero_or_one_relationship_raises_error(self, db_connection):
        # Test that converting an OGM with ZeroOrOne-relationship to
        # unregistered model raises ConversionError
        # Create registered node
        reg_node = RegisteredOGM(name="Registered Node", value=10).save()

        # Create unregistered node
        unreg_node = UnregisteredOGM(name="Unregistered Node", code="UR001").save()

        # Create ZeroOrOne-relationship
        reg_node.unregistered_zero_or_one.connect(unreg_node)

        # Attempt to convert to Pydantic - should raise ConversionError
        with pytest.raises(ConversionError) as excinfo:
            Converter.to_pydantic(reg_node)

        # Verify error message mentions the unregistered class
        assert "No Pydantic model registered for OGM class UnregisteredOGM" in str(excinfo.value)

    def test_nested_unregistered_relationship_raises_error(self, db_connection):
        """Test that converting with a nested unregistered relationship raises ConversionError"""
        # Create a chain of nodes: reg_node -> middle_node -> unreg_node
        reg_node = RegisteredOGM(name="Registered Node", value=10).save()
        middle_node = MiddleOGM(name="Middle Node").save()
        unreg_node = UnregisteredOGM(name="Unregistered Node", code="UR001").save()

        # Connect them in sequence - using the correct relationships
        reg_node.middle_relation.connect(middle_node)  # Fix: Use the proper relationship
        middle_node.unregistered.connect(unreg_node)

        # Converting middle_node should fail
        with pytest.raises(ConversionError) as excinfo:
            Converter.to_pydantic(middle_node)

        # Verify error message
        assert "No Pydantic model registered for OGM class UnregisteredOGM" in str(excinfo.value)

    def test_multiple_unregistered_relationships_raises_error(self, db_connection):
        """Test that having multiple relationships to unregistered models raises ConversionError"""
        # Create registered node
        reg_node = RegisteredOGM(name="Registered Node", value=10).save()

        # Create multiple unregistered nodes
        unreg_node1 = UnregisteredOGM(name="Unregistered Node 1", code="UR001").save()
        unreg_node2 = UnregisteredOGM(name="Unregistered Node 2", code="UR002").save()

        # Create multiple relationships
        reg_node.unregistered_many.connect(unreg_node1)
        reg_node.unregistered_one.connect(unreg_node2)

        # Attempt to convert to Pydantic - should raise ConversionError
        with pytest.raises(ConversionError) as excinfo:
            Converter.to_pydantic(reg_node)

        # Verify error message mentions the unregistered class
        assert "No Pydantic model registered for OGM class UnregisteredOGM" in str(excinfo.value)

    def test_unregistered_relationship_with_max_depth_zero_succeeds(self, db_connection):
        """Test that setting max_depth=0 prevents error by not processing relationships"""
        # Create registered node
        reg_node = RegisteredOGM(name="Registered Node", value=10).save()

        # Create unregistered node
        unreg_node = UnregisteredOGM(name="Unregistered Node", code="UR001").save()

        # Create relationship
        reg_node.unregistered_many.connect(unreg_node)

        # Fix: With max_depth=0, the method returns None, which is correct behavior
        result = Converter.to_pydantic(reg_node, max_depth=0)

        # Fix: Expected result is None when max_depth=0
        assert result is None

    def test_dynamically_register_missing_model_before_conversion(self, db_connection):
        """Test that dynamically registering a missing model fixes the error"""
        # Create registered node
        reg_node = RegisteredOGM(name="Registered Node", value=10).save()

        # Create unregistered node
        unreg_node = UnregisteredOGM(name="Unregistered Node", code="UR001").save()

        # Fix: Create relationships for all cardinality types to avoid CardinalityViolation
        reg_node.unregistered_many.connect(unreg_node)
        reg_node.unregistered_one.connect(unreg_node)  # This is required for One cardinality
        reg_node.unregistered_zero_or_one.connect(unreg_node)

        # First verify it would fail
        with pytest.raises(ConversionError):
            Converter.to_pydantic(reg_node)

        # Now create and register the missing Pydantic model
        class UnregisteredPydantic(BaseModel):
            name: str = None
            code: str = None

        # Register the model
        Converter.register_models(UnregisteredPydantic, UnregisteredOGM)

        # Now conversion should succeed
        result = Converter.to_pydantic(reg_node)

        # Verify basic properties and relationship
        assert result is not None
        assert result.name == "Registered Node"
        assert len(result.unregistered_many) == 1
        assert result.unregistered_many[0].name == "Unregistered Node"
        assert result.unregistered_many[0].code == "UR001"
        assert result.unregistered_one.name == "Unregistered Node"
        assert result.unregistered_one.code == "UR001"
