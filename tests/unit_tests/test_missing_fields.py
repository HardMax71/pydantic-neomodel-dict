from typing import Optional

import pytest
from neomodel import BooleanProperty, IntegerProperty, StringProperty, StructuredNode
from pydantic import BaseModel

from pydantic_neomodel_dict import Converter

# ===== Module-level model definitions =====

class MinimalUserPydantic(BaseModel):
    name: str
    email: str
    # Missing fields: age, created_at, is_active


class FullUserPydantic(MinimalUserPydantic):
    age: Optional[int] = None
    created_at: Optional[str] = None
    is_active: Optional[bool] = None


class ExtendedUserOGM(StructuredNode):
    name = StringProperty(required=True)
    email = StringProperty(unique_index=True)
    age = IntegerProperty(default=0)  # Extra field
    created_at = StringProperty()  # Extra field
    is_active = BooleanProperty(default=True)  # Extra field


# ===== Fixtures =====

@pytest.fixture
def registered_models():
    """Register models"""
    Converter.register_models(MinimalUserPydantic, ExtendedUserOGM)
    Converter.register_models(FullUserPydantic, ExtendedUserOGM)
    yield


@pytest.fixture
def minimal_user():
    """Create a minimal user"""
    return MinimalUserPydantic(name="Test User", email="test@example.com")


# ===== Test Class =====

class TestMissingFields:
    """Tests for handling missing fields between models"""

    def test_missing_pydantic_fields(self, db_connection, registered_models, minimal_user):
        """
        Test handling of OGM models with fields not in Pydantic models.

        Verifies that conversion succeeds when OGM has extra fields not present
        in Pydantic, and that extended Pydantic models can capture these fields.
        """
        # Get user from fixture
        user_pydantic = minimal_user

        # Convert to OGM
        user_ogm = Converter.to_ogm(user_pydantic)

        # Set extra fields in OGM
        user_ogm.age = 25
        user_ogm.created_at = "2023-01-01"
        user_ogm.is_active = True

        # Verify properties
        assert user_ogm.name == "Test User", "Name not preserved"
        assert user_ogm.email == "test@example.com", "Email not preserved"
        assert user_ogm.age == 25, "Age not set correctly"

        # Convert back to minimal Pydantic model
        converted_back = Converter.to_pydantic(user_ogm, MinimalUserPydantic)

        # Verify only matching properties are preserved
        assert converted_back.name == user_pydantic.name, "Name not preserved in round trip"
        assert converted_back.email == user_pydantic.email, "Email not preserved in round trip"

        # Verify extra fields are not included
        with pytest.raises(AttributeError):
            assert converted_back.age

        # Convert using the extended Pydantic class
        full_user = Converter.to_pydantic(user_ogm, FullUserPydantic)

        # Verify all fields are preserved
        assert full_user.name == "Test User", "Name not preserved in extended model"
        assert full_user.email == "test@example.com", "Email not preserved in extended model"
        assert full_user.age == 25, "Age not preserved in extended model"
        assert full_user.created_at == "2023-01-01", "created_at not preserved in extended model"
        assert full_user.is_active is True, "is_active not preserved in extended model"
