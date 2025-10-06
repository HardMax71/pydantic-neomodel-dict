import pytest
from neomodel import IntegerProperty, StringProperty, StructuredNode
from pydantic import BaseModel

from pydantic_neomodel_dict.converters import SyncConverter

Converter = SyncConverter()

# ===== Module-level model definitions =====

class UserPydantic(BaseModel):
    name: str
    email: str
    age: int


class SimpleTestUserOGM(StructuredNode):
    name = StringProperty(required=True)
    email = StringProperty(unique_index=True, required=True)
    age = IntegerProperty(index=True, default=0)


# ===== Fixtures =====

@pytest.fixture
def registered_models():
    """Register models"""
    Converter.register_models(UserPydantic, SimpleTestUserOGM)
    yield


@pytest.fixture
def user_fixture():
    """Create a test user"""
    return UserPydantic(name="John Doe", email="john@example.com", age=30)


# ===== Test Class =====

class TestSimpleConversion:
    """Tests for simple model conversion with basic properties"""

    def test_simple_conversion(self, db_connection, registered_models, user_fixture):
        """
        Test converting a basic model with simple properties between Pydantic and OGM.

        This verifies that primitive types are correctly preserved during conversion
        in both directions.
        """
        # Get user from fixture
        user_pydantic = user_fixture

        # Convert to OGM
        user_ogm = Converter.to_ogm(user_pydantic)

        # Verify properties were preserved
        assert user_ogm.name == "John Doe", "Name not preserved in conversion to OGM"
        assert user_ogm.email == "john@example.com", "Email not preserved in conversion to OGM"
        assert user_ogm.age == 30, "Age not preserved in conversion to OGM"

        # Convert back to Pydantic
        converted_back = Converter.to_pydantic(user_ogm)

        # Verify properties were preserved in the round trip
        assert converted_back.name == user_pydantic.name, "Name not preserved in round trip"
        assert converted_back.email == user_pydantic.email, "Email not preserved in round trip"
        assert converted_back.age == user_pydantic.age, "Age not preserved in round trip"
