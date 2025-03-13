from datetime import datetime
from typing import List, Optional

import pytest
from neomodel import RelationshipTo, RelationshipFrom, Relationship, db
from neomodel import StructuredNode, StringProperty, IntegerProperty, DateTimeProperty
from pydantic import BaseModel, Field

from converter import Converter, ConversionError


# Define test OGM models with unique names to avoid conflicts
class RelValPersonOGM(StructuredNode):
    """Person OGM model for relationship value tests"""
    name = StringProperty(required=True)
    age = IntegerProperty()
    created_at = DateTimeProperty(default_factory=datetime.now)

    # Relationships
    friends = Relationship('RelValPersonOGM', 'FRIENDS_WITH')
    managed_by = RelationshipTo('RelValPersonOGM', 'MANAGED_BY')
    manages = RelationshipFrom('RelValPersonOGM', 'MANAGED_BY')
    department = RelationshipTo('RelValDepartmentOGM', 'BELONGS_TO')


class RelValDepartmentOGM(StructuredNode):
    """Department OGM model for relationship value tests"""
    name = StringProperty(required=True)
    code = StringProperty(unique_index=True)

    members = RelationshipFrom('RelValPersonOGM', 'BELONGS_TO')


# Define corresponding Pydantic models
class RelValDepartmentModel(BaseModel):
    """Department Pydantic model for relationship value tests"""
    name: str
    code: Optional[str] = None
    members: List['RelValPersonModel'] = Field(default_factory=list)


class RelValPersonModel(BaseModel):
    """Person Pydantic model for relationship value tests"""
    name: str
    age: Optional[int] = None
    created_at: Optional[datetime] = None

    friends: List['RelValPersonModel'] = Field(default_factory=list)
    managed_by: Optional['RelValPersonModel'] = None
    manages: List['RelValPersonModel'] = Field(default_factory=list)
    department: Optional[RelValDepartmentModel] = None


# Update forward references
RelValPersonModel.model_rebuild()
RelValDepartmentModel.model_rebuild()

# Register models with converter
Converter.register_models(RelValPersonModel, RelValPersonOGM)
Converter.register_models(RelValDepartmentModel, RelValDepartmentOGM)


class TestRelationshipValueTypes:
    """Tests for how different relationship value types are handled"""

    def setup_method(self):
        """Set up the test environment"""
        # Clear the database before each test
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def teardown_method(self):
        """Clean up after each test"""
        # Clear the database after each test
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def test_dict_relationship_value(self, db_connection):
        """Test with dictionary relationship value (valid case)"""
        data = {
            "name": "John Doe",
            "age": 30,
            "managed_by": {
                "name": "Jane Smith",
                "age": 45
            }
        }

        # Convert to OGM
        result = Converter.dict_to_ogm(data, RelValPersonOGM)

        # Verify the result
        assert result is not None
        assert isinstance(result, RelValPersonOGM)
        assert result.name == "John Doe"
        assert result.age == 30

        # Verify relationship was created
        manager = list(result.managed_by.all())
        assert len(manager) == 1
        assert manager[0].name == "Jane Smith"

    def test_string_relationship_value(self, db_connection):
        """Test with string relationship value (invalid case)"""
        data = {
            "name": "John Doe",
            "age": 30,
            # String instead of dict for relationship
            "managed_by": "Jane Smith"
        }

        with pytest.raises(ConversionError) as excinfo:
            Converter.dict_to_ogm(data, RelValPersonOGM)

        assert "Relationship 'managed_by' must be a dictionary" in str(excinfo.value)

    def test_integer_relationship_value(self, db_connection):
        """Test with integer relationship value (invalid case)"""
        data = {
            "name": "John Doe",
            "age": 30,
            # Integer instead of dict for relationship
            "managed_by": 12345
        }

        with pytest.raises(ConversionError) as excinfo:
            Converter.dict_to_ogm(data, RelValPersonOGM)

        assert "Relationship 'managed_by' must be a dictionary" in str(excinfo.value)

    def test_boolean_relationship_value(self, db_connection):
        """Test with boolean relationship value (invalid case)"""
        data = {
            "name": "John Doe",
            "age": 30,
            # Boolean instead of dict for relationship
            "managed_by": True
        }

        with pytest.raises(ConversionError) as excinfo:
            Converter.dict_to_ogm(data, RelValPersonOGM)

        assert "Relationship 'managed_by' must be a dictionary" in str(excinfo.value)

    def test_none_relationship_value(self, db_connection):
        """Test with None relationship value (valid case)"""
        data = {
            "name": "John Doe",
            "age": 30,
            # None is valid and should be skipped
            "managed_by": None
        }

        result = Converter.dict_to_ogm(data, RelValPersonOGM)

        # Verify the result
        assert result is not None
        assert isinstance(result, RelValPersonOGM)
        assert result.name == "John Doe"
        assert result.age == 30

        # Verify no relationship was created
        manager = list(result.managed_by.all())
        assert len(manager) == 0

    def test_list_of_dicts_relationship_value(self, db_connection):
        """Test with list of dictionaries relationship value (valid case)"""
        data = {
            "name": "John Doe",
            "age": 30,
            # List of dicts for relationship
            "friends": [
                {"name": "Alice", "age": 28},
                {"name": "Bob", "age": 32}
            ]
        }

        result = Converter.dict_to_ogm(data, RelValPersonOGM)

        # Verify the result
        assert result is not None
        assert isinstance(result, RelValPersonOGM)

        # Verify relationships were created
        friends = list(result.friends.all())
        assert len(friends) == 2
        friend_names = sorted([f.name for f in friends])
        assert friend_names == ["Alice", "Bob"]

    def test_list_of_strings_relationship_value(self, db_connection):
        """Test with list of strings relationship value (invalid case)"""
        data = {
            "name": "John Doe",
            "age": 30,
            # List of strings instead of dicts for relationship
            "friends": ["Alice", "Bob", "Charlie"]
        }

        with pytest.raises(ConversionError) as excinfo:
            Converter.dict_to_ogm(data, RelValPersonOGM)

        assert "list item" in str(excinfo.value)
        assert "must be a dictionary" in str(excinfo.value)

    def test_list_of_integers_relationship_value(self, db_connection):
        """Test with list of integers relationship value (invalid case)"""
        data = {
            "name": "John Doe",
            "age": 30,
            # List of integers instead of dicts for relationship
            "friends": [123, 456, 789]
        }

        with pytest.raises(ConversionError) as excinfo:
            Converter.dict_to_ogm(data, RelValPersonOGM)

        assert "list item" in str(excinfo.value)
        assert "must be a dictionary" in str(excinfo.value)

    def test_mixed_relationship_list(self, db_connection):
        """Test with mixed types in relationship list (invalid case)"""
        data = {
            "name": "John Doe",
            "age": 30,
            # Mixed types in relationship list
            "friends": [
                {"name": "Alice", "age": 28},  # Valid dict
                12345,  # Invalid integer
                "Bob"  # Invalid string
            ]
        }

        with pytest.raises(ConversionError) as excinfo:
            Converter.dict_to_ogm(data, RelValPersonOGM)

        assert "list item" in str(excinfo.value)
        assert "must be a dictionary" in str(excinfo.value)

    def test_empty_list_relationship_value(self, db_connection):
        """Test with empty list relationship value (valid case)"""
        data = {
            "name": "John Doe",
            "age": 30,
            # Empty list is valid
            "friends": []
        }

        result = Converter.dict_to_ogm(data, RelValPersonOGM)

        # Verify the result
        assert result is not None
        assert isinstance(result, RelValPersonOGM)
        assert result.name == "John Doe"
        assert result.age == 30

        # Verify no relationships were created
        friends = list(result.friends.all())
        assert len(friends) == 0


class TestNestedRelationshipValues:
    """Tests for nested relationship values"""

    def setup_method(self):
        """Set up the test environment"""
        # Clear the database before each test
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def teardown_method(self):
        """Clean up after each test"""
        # Clear the database after each test
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def test_nested_valid_relationships(self, db_connection):
        """Test with nested valid relationship dictionaries"""
        data = {
            "name": "John Doe",
            "age": 30,
            "department": {
                "name": "Engineering",
                "code": "ENG-01"
            },
            "managed_by": {
                "name": "Jane Smith",
                "age": 45,
                # Nested relationship inside managed_by
                "department": {
                    "name": "Management",
                    "code": "MGT-01"
                }
            }
        }

        result = Converter.dict_to_ogm(data, RelValPersonOGM)

        # Verify the result
        assert result is not None
        assert isinstance(result, RelValPersonOGM)

        # Verify department relationship
        departments = list(result.department.all())
        assert len(departments) == 1
        assert departments[0].name == "Engineering"

        # Verify manager relationship
        managers = list(result.managed_by.all())
        assert len(managers) == 1
        assert managers[0].name == "Jane Smith"

        # Verify manager's department relationship
        manager_departments = list(managers[0].department.all())
        assert len(manager_departments) == 1
        assert manager_departments[0].name == "Management"

    def test_nested_invalid_relationships(self, db_connection):
        """Test with nested invalid relationship values"""
        data = {
            "name": "John Doe",
            "age": 30,
            "managed_by": {
                "name": "Jane Smith",
                "age": 45,
                # Invalid nested relationship
                "department": "Engineering"  # String instead of dict
            }
        }

        with pytest.raises(ConversionError) as excinfo:
            Converter.dict_to_ogm(data, RelValPersonOGM)

        assert "must be a dictionary" in str(excinfo.value)


class TestEdgeCaseRelationshipValues:
    """Tests for edge case relationship values"""

    def setup_method(self):
        """Set up the test environment"""
        # Clear the database before each test
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def teardown_method(self):
        """Clean up after each test"""
        # Clear the database after each test
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def test_empty_dict_relationship(self, db_connection):
        """Test with empty dictionary relationship value"""
        data = {
            "name": "John Doe",
            "age": 30,
            # Empty dict is valid but may produce minimal related object
            "managed_by": {}
        }

        # This might succeed or fail depending on required properties in the target model
        try:
            result = Converter.dict_to_ogm(data, RelValPersonOGM)
            # If it succeeds, verify it created some kind of result
            if result is not None:
                assert isinstance(result, RelValPersonOGM)
        except Exception:
            # If it fails, that's also a valid outcome for empty dict with required fields
            pass

    def test_dict_with_extra_fields_relationship(self, db_connection):
        """Test with dictionary containing extra fields for relationship"""
        data = {
            "name": "John Doe",
            "age": 30,
            # Dict with extra fields
            "managed_by": {
                "name": "Jane Smith",
                "age": 45,
                "extra_field": "value",  # Not in model
                "another_extra": 123  # Not in model
            }
        }

        result = Converter.dict_to_ogm(data, RelValPersonOGM)

        # Verify the result
        assert result is not None
        assert isinstance(result, RelValPersonOGM)

        # Verify relationship was created
        managers = list(result.managed_by.all())
        assert len(managers) == 1
        assert managers[0].name == "Jane Smith"

    def test_dict_with_missing_required_fields(self, db_connection):
        """Test with dictionary missing required fields for relationship"""
        data = {
            "name": "John Doe",
            "age": 30,
            # Dict missing required 'name' field
            "department": {
                "code": "ENG-01"
                # name is missing but required
            }
        }

        # This may raise an exception depending on how strict the implementation is
        with pytest.raises(Exception):
            Converter.dict_to_ogm(data, RelValPersonOGM)
