import pytest
from neomodel import StructuredNode, StringProperty, IntegerProperty
from pydantic import BaseModel

from converter import Converter


# ===== Module-level model definitions =====

class UserPydantic(BaseModel):
    name: str
    email: str
    age: int


class UserOGM_CustomType(StructuredNode):
    name = StringProperty(required=True)
    email = StringProperty(unique_index=True, required=True)
    age = IntegerProperty(index=True, default=0)


# ===== Fixtures =====

@pytest.fixture
def registered_models():
    """Register models"""
    Converter.register_models(UserPydantic, UserOGM_CustomType)
    yield


@pytest.fixture
def user_batch():
    """Create a batch of users"""
    return [
        UserPydantic(name=f"User{i}", email=f"user{i}@example.com", age=20 + i)
        for i in range(1, 6)
    ]


# ===== Test Class =====

class TestBatchConversion:
    """Tests for batch conversion operations"""

    def test_batch_conversion(self, db_connection, registered_models, user_batch):
        """
        Test batch conversion for multiple instances at once.

        Verifies that batch conversion works efficiently for multiple objects,
        preserving all properties in a single transaction.
        """
        # Get users from fixture
        users = user_batch

        # Batch convert to OGM
        user_ogms = Converter.batch_to_ogm(users)

        # Verify all users were converted
        assert len(user_ogms) == 5, "Not all users were converted"

        # Verify properties for each user
        for i, user in enumerate(user_ogms):
            assert user.name == f"User{i + 1}", f"Name not preserved for user {i + 1}"
            assert user.email == f"user{i + 1}@example.com", f"Email not preserved for user {i + 1}"
            assert user.age == 20 + (i + 1), f"Age not preserved for user {i + 1}"

        # Batch convert back to Pydantic
        converted_back = Converter.batch_to_pydantic(user_ogms)

        # Verify all users were converted back
        assert len(converted_back) == 5, "Not all users were converted back"

        # Verify properties were preserved in the round trip
        for i, user in enumerate(converted_back):
            assert user.name == f"User{i + 1}", f"Name not preserved in round trip for user {i + 1}"
            assert user.email == f"user{i + 1}@example.com", f"Email not preserved in round trip for user {i + 1}"
            assert user.age == 20 + (i + 1), f"Age not preserved in round trip for user {i + 1}"
