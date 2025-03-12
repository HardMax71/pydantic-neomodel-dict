from typing import List

import pytest
from neomodel import StructuredNode, StringProperty, RelationshipTo
from pydantic import BaseModel, Field

from converter import Converter


# ===== Module-level model definitions =====

class FriendPydantic(BaseModel):
    name: str
    email: str
    friends: List['FriendPydantic'] = Field(default_factory=list)


# Resolve forward reference
FriendPydantic.model_rebuild()


class FriendOGM(StructuredNode):
    name = StringProperty(required=True)
    email = StringProperty(unique_index=True)
    friends = RelationshipTo('FriendOGM', 'IS_FRIEND_WITH')


# ===== Fixtures =====

@pytest.fixture
def registered_models():
    """Register models"""
    Converter.register_models(FriendPydantic, FriendOGM)
    yield


@pytest.fixture
def cyclic_friend_network():
    """Create a network of friends with cyclic references"""
    # Create a network with cyclic references
    alice = FriendPydantic(name="Alice", email="alice@example.com")
    bob = FriendPydantic(name="Bob", email="bob@example.com")
    charlie = FriendPydantic(name="Charlie", email="charlie@example.com")

    # Create cycle: Alice -> Bob -> Charlie -> Alice
    alice.friends = [bob, charlie]
    bob.friends = [alice, charlie]
    charlie.friends = [alice, bob]

    return alice


# ===== Test Class =====

class TestCyclicReferences:
    """Tests for cyclic references between models"""

    def test_cyclic_references(self, db_connection, registered_models, cyclic_friend_network):
        """
        Test handling of cyclic references in models.

        Verifies that circular references (A->B->C->A) are properly handled
        without causing infinite recursion.
        """
        # Get the cyclic network from fixture
        alice = cyclic_friend_network

        # Convert to OGM
        alice_ogm = Converter.to_ogm(alice)

        # Verify properties
        assert alice_ogm.name == "Alice", "Name not preserved"
        assert alice_ogm.email == "alice@example.com", "Email not preserved"

        # Verify relationships
        friends = list(alice_ogm.friends.all())
        assert len(friends) == 2, "Incorrect number of friends"

        friend_names = sorted([friend.name for friend in friends])
        assert friend_names == ["Bob", "Charlie"], "Friend names not preserved"

        # Convert back to Pydantic
        converted_back = Converter.to_pydantic(alice_ogm)

        # Verify properties
        assert converted_back.name == alice.name, "Name not preserved in round trip"
        assert converted_back.email == alice.email, "Email not preserved in round trip"
        assert len(converted_back.friends) == 2, "Incorrect number of friends in round trip"

        # Verify cyclic references were maintained
        friend_names = sorted([f.name for f in converted_back.friends])
        assert friend_names == ["Bob", "Charlie"], "Friend names not preserved in round trip"

        # Find Bob in the friends
        bob_converted = next((f for f in converted_back.friends if f.name == "Bob"), None)
        assert bob_converted is not None, "Bob not found in friends"

        # Verify Bob's friends includes Alice (confirming cycle was preserved)
        bob_friend_names = [f.name for f in bob_converted.friends]
        assert "Alice" in bob_friend_names, "Cyclic reference to Alice not preserved"
