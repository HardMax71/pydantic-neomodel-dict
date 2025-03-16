from datetime import date
from typing import List, Optional

import pytest
from neomodel import (
    BooleanProperty,
    DateProperty,
    FloatProperty,
    IntegerProperty,
    RelationshipTo,
    StringProperty,
    StructuredNode,
    ZeroOrOne,
)
from pydantic import BaseModel, Field

from pydantic_neo4j_dict import ConversionError, Converter

# ===== Module-level model definitions =====

# Models for direct conversion test
class DirectPydantic(BaseModel):
    name: str
    value: int


class DirectOGM(StructuredNode):
    name = StringProperty()
    value = IntegerProperty()


# Models for missing fields test
class MissingFieldsPydantic(BaseModel):
    id: Optional[str] = None  # Only in Pydantic
    name: str  # Common field


class MissingFieldsOGM(StructuredNode):
    name = StringProperty()  # Common field
    count = IntegerProperty(default=0)  # Only in OGM


# Models for None relationship test
class RelatedPydantic(BaseModel):
    name: str


class ParentPydantic(BaseModel):
    name: str
    related: Optional[RelatedPydantic] = None


class RelatedOGM(StructuredNode):
    name = StringProperty()


class ParentOGM(StructuredNode):
    name = StringProperty()
    related = RelationshipTo(RelatedOGM, 'HAS_RELATED', cardinality=ZeroOrOne)


# Models for cyclic references test
class CyclicPydantic(BaseModel):
    name: str
    links: List['CyclicPydantic'] = Field(default_factory=list)


# Resolve forward references
CyclicPydantic.model_rebuild()


class CyclicOGM(StructuredNode):
    name = StringProperty()
    links = RelationshipTo('CyclicOGM', 'LINKS_TO')


# Models for complex nested structure test
class ItemPydantic(BaseModel):
    name: str
    price: float


class OrderPydantic(BaseModel):
    uid: str
    items: List[ItemPydantic] = Field(default_factory=list)


class CustomerPydantic(BaseModel):
    name: str
    email: str
    orders: List[OrderPydantic] = Field(default_factory=list)


class ItemOGM(StructuredNode):
    name = StringProperty()
    price = FloatProperty()


class OrderOGM(StructuredNode):
    uid = StringProperty()
    items = RelationshipTo(ItemOGM, 'CONTAINS')


class CustomerOGM(StructuredNode):
    name = StringProperty()
    email = StringProperty()
    orders = RelationshipTo(OrderOGM, 'PLACED')


# Models for type conversion test
class TypesPydantic(BaseModel):
    string_val: str
    int_val: int
    float_val: float
    bool_val: bool
    date_val: date


class TypesOGM(StructuredNode):
    string_val = StringProperty()
    int_val = IntegerProperty()
    float_val = FloatProperty()
    bool_val = BooleanProperty()
    date_val = DateProperty()


# Models for error cases test
class RequiredFieldsPydantic(BaseModel):
    name: str  # Required with no default


class UnregisteredPydantic(BaseModel):
    field: str


# ===== Fixtures =====

@pytest.fixture
def register_cyclic_models():
    """Register cyclic models"""
    Converter.register_models(CyclicPydantic, CyclicOGM)
    yield


@pytest.fixture
def direct_pydantic():
    """Create direct test instance"""
    return DirectPydantic(name="DirectTest", value=42)


@pytest.fixture
def missing_fields_pydantic():
    """Create instance with missing fields"""
    return MissingFieldsPydantic(id="test123", name="TestName")


@pytest.fixture
def none_parent():
    """Create parent with None relationship"""
    return ParentPydantic(name="Parent", related=None)


@pytest.fixture
def some_parent():
    """Create parent with relationship"""
    return ParentPydantic(name="Parent2", related=RelatedPydantic(name="Child"))


@pytest.fixture
def cyclic_references():
    """Create cycle: A->B->C->A"""
    cycle_a = CyclicPydantic(name="NodeA")
    cycle_b = CyclicPydantic(name="NodeB")
    cycle_c = CyclicPydantic(name="NodeC")

    cycle_a.links = [cycle_b]
    cycle_b.links = [cycle_c]
    cycle_c.links = [cycle_a]  # Creates cycle

    return cycle_a


@pytest.fixture
def complex_customer():
    """Create complex nested structure"""
    item1 = ItemPydantic(name="Product1", price=10.99)
    item2 = ItemPydantic(name="Product2", price=24.99)

    order = OrderPydantic(uid="ORD-123", items=[item1, item2])

    return CustomerPydantic(
        name="Nested Customer",
        email="nested@example.com",
        orders=[order]
    )


@pytest.fixture
def types_data():
    """Create instance with values needing conversion"""
    today = date.today()
    return TypesPydantic(
        string_val="42",  # Will stay string
        int_val=42.0,  # Will convert float->int
        float_val="42.5",  # Will convert string->float
        bool_val=0,  # Will convert int->bool
        date_val=today  # Will convert date->string
    )


@pytest.fixture
def register_type_converters():
    """Register date converters"""
    Converter.register_type_converter(
        date, str, lambda d: d.isoformat()
    )
    Converter.register_type_converter(
        str, date, lambda s: date.fromisoformat(s)
    )
    yield


# ===== Test Class =====

class TestComprehensive:
    """Comprehensive tests covering multiple edge cases"""

    def test_direct_conversion(self, db_connection, direct_pydantic):
        """
        Test direct conversion without registration.

        Verifies that conversion works when explicitly providing model classes
        without prior registration.
        """
        # Get instance from fixture
        direct_pydantic_instance = direct_pydantic

        # Convert directly without prior registration
        direct_ogm = Converter.to_ogm(direct_pydantic_instance, DirectOGM)

        # Verify properties
        assert direct_ogm.name == "DirectTest", "Name not preserved in direct conversion"
        assert direct_ogm.value == 42, "Value not preserved in direct conversion"

        # Convert back with explicit class
        back_pydantic = Converter.to_pydantic(direct_ogm, DirectPydantic)

        # Verify properties
        assert back_pydantic.name == direct_pydantic_instance.name, "Name not preserved in direct round trip"
        assert back_pydantic.value == direct_pydantic_instance.value, "Value not preserved in direct round trip"

    def test_missing_fields_both_directions(self, db_connection, missing_fields_pydantic):
        """
        Test handling of missing fields in both directions.

        Verifies that conversion succeeds when fields are missing in either
        the Pydantic or OGM model.
        """
        # Get instance from fixture
        missing_pydantic = missing_fields_pydantic

        # Convert to OGM
        missing_ogm = Converter.to_ogm(missing_pydantic, MissingFieldsOGM)

        # Verify common field preserved
        assert missing_ogm.name == "TestName", "Common field not preserved"

        # Set OGM-only field
        missing_ogm.count = 99

        # Convert back
        back_missing = Converter.to_pydantic(missing_ogm, MissingFieldsPydantic)

        # Verify common field preserved, OGM-only field lost, Pydantic-only field preserved as default
        assert back_missing.name == "TestName", "Common field not preserved in round trip"
        assert back_missing.id is None, "Pydantic-only field should be None (default)"

    def test_none_relationship_handling(self, db_connection, none_parent, some_parent):
        """
        Test comprehensive handling of None in relationships.

        Verifies that None values in relationships are properly handled
        across conversion operations.
        """
        # Convert parent with None related to OGM
        parent_ogm = Converter.to_ogm(none_parent, ParentOGM)

        # Verify relationship empty
        assert len(list(parent_ogm.related.all())) == 0, "None relationship should result in no connections"

        # Convert parent with related to OGM
        parent_ogm2 = Converter.to_ogm(some_parent, ParentOGM)

        # Verify relationship created
        related_nodes = list(parent_ogm2.related.all())
        assert len(related_nodes) == 1, "Relationship not created"
        assert related_nodes[0].name == "Child", "Related node property not preserved"

    def test_cyclic_references_explicit(self, db_connection, register_cyclic_models, cyclic_references):
        """
        Test handling of cyclic references with explicit model specification.

        Verifies that circular references are properly handled with direct conversion.
        """
        # Get cycle from fixture
        cycle_a = cyclic_references

        # Convert with cycle
        cycle_ogm_a = Converter.to_ogm(cycle_a)

        # Verify properties and relationships
        assert cycle_ogm_a.name == "NodeA", "Node name not preserved"

        # Check cycle was created
        ogm_bs = list(cycle_ogm_a.links.all())
        assert len(ogm_bs) == 1, "First link not created"
        assert ogm_bs[0].name == "NodeB", "First link target not preserved"

        ogm_cs = list(ogm_bs[0].links.all())
        assert len(ogm_cs) == 1, "Second link not created"
        assert ogm_cs[0].name == "NodeC", "Second link target not preserved"

        ogm_as = list(ogm_cs[0].links.all())
        assert len(ogm_as) == 1, "Third link not created"
        assert ogm_as[0].name == "NodeA", "Cycle not complete"

        # Convert back
        cycle_back = Converter.to_pydantic(cycle_ogm_a)

        # Verify cycle preserved
        assert cycle_back.name == "NodeA", "Node name not preserved in round trip"
        assert len(cycle_back.links) == 1, "Link not preserved in round trip"
        assert cycle_back.links[0].name == "NodeB", "Link target not preserved in round trip"

    def test_complex_nested_structure(self, db_connection, complex_customer):
        """
        Test handling of complex nested structures.

        Verifies conversion of multi-level nested structures works correctly.
        """
        # Get complex structure from fixture
        customer = complex_customer

        # Convert
        customer_ogm = Converter.to_ogm(customer, CustomerOGM)

        # Verify structure
        assert customer_ogm.name == "Nested Customer", "Customer name not preserved"

        orders = list(customer_ogm.orders.all())
        assert len(orders) == 1, "Order not created"
        assert orders[0].uid == "ORD-123", "Order ID not preserved"

        items = list(orders[0].items.all())
        assert len(items) == 2, "Items not created"
        assert items[0].name in ["Product1", "Product2"], "Item name not preserved"
        assert items[1].name in ["Product1", "Product2"], "Item name not preserved"
        assert items[0].name != items[1].name, "Items not distinct"

        # Test prices - get them in the same order as the items
        item_prices = {item.name: item.price for item in items}
        assert item_prices["Product1"] == 10.99, "Item price not preserved"
        assert item_prices["Product2"] == 24.99, "Item price not preserved"

    def test_type_conversions(self, db_connection, register_type_converters, types_data):
        """
        Test automatic type conversion between different data types.

        Verifies that values are automatically converted between compatible types.
        """
        # Get types data from fixture
        types_pydantic = types_data
        today = types_pydantic.date_val

        # Convert to OGM
        types_ogm = Converter.to_ogm(types_pydantic, TypesOGM)

        # Verify conversions
        assert types_ogm.string_val == "42", "String value not preserved"
        assert types_ogm.int_val == 42, "Float not converted to int"
        assert types_ogm.float_val == 42.5, "String not converted to float"
        assert types_ogm.bool_val is False, "Int not converted to bool"
        assert types_ogm.date_val == today, "Date not preserved as date object"

        # Verify types
        assert isinstance(types_ogm.string_val, str), "string_val type incorrect"
        assert isinstance(types_ogm.int_val, int), "int_val type incorrect"
        assert isinstance(types_ogm.float_val, float), "float_val type incorrect"
        assert isinstance(types_ogm.bool_val, bool), "bool_val type incorrect"
        assert isinstance(types_ogm.date_val, date), "date_val type incorrect"

    def test_error_cases(self, db_connection):
        """
        Test various error conditions are handled properly.

        Verifies that appropriate errors are raised for invalid operations.
        """
        # Test 1: Missing required field
        with pytest.raises(Exception) as excinfo:
            # Should fail validation
            RequiredFieldsPydantic()

        # Verify error contains field name
        assert "name" in str(excinfo.value).lower()

        # Test 2: Convert unregistered class without target
        with pytest.raises(ConversionError) as excinfo:
            # Should fail with no OGM class specified or registered
            Converter.to_ogm(UnregisteredPydantic(field="test"))

        # Verify error message
        assert "No mapping registered" in str(excinfo.value), "Error should mention missing registration"
