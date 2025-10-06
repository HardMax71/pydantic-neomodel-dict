from typing import Optional

import pytest
from neomodel import RelationshipTo, StringProperty, StructuredNode, ZeroOrOne
from pydantic import BaseModel

from pydantic_neomodel_dict.converters import SyncConverter

Converter = SyncConverter()

# ===== Module-level model definitions =====

class AddressPydantic(BaseModel):
    street: str
    city: str
    zip_code: str


class PersonPydantic(BaseModel):
    name: str
    email: str
    address: Optional[AddressPydantic] = None


class AddressRelationshipsOGM(StructuredNode):
    street = StringProperty(required=True)
    city = StringProperty(required=True)
    zip_code = StringProperty(required=True)


class PersonRelationshipsOGM(StructuredNode):
    name = StringProperty(required=True)
    email = StringProperty(unique_index=True)
    address = RelationshipTo(AddressRelationshipsOGM, 'HAS_ADDRESS', cardinality=ZeroOrOne)


# ===== Fixtures =====

@pytest.fixture
def registered_models():
    """Register models"""
    Converter.register_models(AddressPydantic, AddressRelationshipsOGM)
    Converter.register_models(PersonPydantic, PersonRelationshipsOGM)
    yield


@pytest.fixture
def person_with_address():
    """Create a person with an address"""
    address = AddressPydantic(
        street="123 Main St",
        city="New York",
        zip_code="10001"
    )
    person = PersonPydantic(
        name="Jane Smith",
        email="jane@example.com",
        address=address
    )
    return person


# ===== Test Class =====

class TestRelationships:
    """Tests for models with relationships"""

    def test_single_relationship(self, db_connection, registered_models, person_with_address):
        """
        Test converting models with a single relationship between entities.

        Verifies that relationships are properly established and preserved
        during conversion in both directions.
        """
        # Get person from fixture
        person = person_with_address
        address = person.address

        # Convert to OGM
        person_ogm = Converter.to_ogm(person)

        # Verify person properties
        assert person_ogm.name == "Jane Smith", "Person name not preserved"
        assert person_ogm.email == "jane@example.com", "Person email not preserved"

        # Verify relationship was properly created
        address_related = list(person_ogm.address.all())
        assert len(address_related) == 1, "Address relationship not created"
        assert address_related[0].street == "123 Main St", "Address street not preserved"
        assert address_related[0].city == "New York", "Address city not preserved"
        assert address_related[0].zip_code == "10001", "Address zip_code not preserved"

        # Convert back to Pydantic
        converted_back = Converter.to_pydantic(person_ogm)

        # Verify properties
        assert converted_back.name == person.name, "Person name not preserved in round trip"
        assert converted_back.email == person.email, "Person email not preserved in round trip"

        # Verify relationship was preserved
        assert converted_back.address is not None, "Address relationship lost in round trip"
        assert converted_back.address.street == address.street, "Address street not preserved in round trip"
        assert converted_back.address.city == address.city, "Address city not preserved in round trip"
        assert converted_back.address.zip_code == address.zip_code, "Address zip_code not preserved in round trip"
