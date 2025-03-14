import pytest
from neomodel import StructuredNode, StringProperty, IntegerProperty, FloatProperty, BooleanProperty, RelationshipTo

from converter import Converter


# ===== Module-level model definitions =====

class UserOGM_DictTest(StructuredNode):
    name = StringProperty(required=True)
    email = StringProperty(unique_index=True)
    age = IntegerProperty(default=25)


class ProductOGM_DictTest(StructuredNode):
    name = StringProperty(required=True)
    price = FloatProperty(default=0.0)
    in_stock = BooleanProperty(default=True)


class AddressOGM_DictTest(StructuredNode):
    street = StringProperty(required=True)
    city = StringProperty(required=True)


class CustomerOGM_DictTest(StructuredNode):
    name = StringProperty(required=True)
    addresses = RelationshipTo(AddressOGM_DictTest, 'HAS_ADDRESS')


class NodeOGM_DictTest(StructuredNode):
    name = StringProperty(required=True)
    links_to = RelationshipTo('NodeOGM_DictTest', 'LINKS_TO')


class ItemOGM_DictTest(StructuredNode):
    name = StringProperty(required=True)
    quantity = IntegerProperty(default=0)


class TagOGM_DictTest(StructuredNode):
    name = StringProperty(required=True)


class ArticleOGM_DictTest(StructuredNode):
    title = StringProperty(required=True)
    content = StringProperty()
    tags = RelationshipTo(TagOGM_DictTest, 'HAS_TAG')


# ===== Fixtures =====

@pytest.fixture
def user_dict():
    """Create a user dictionary"""
    return {
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30
    }


@pytest.fixture
def user_dict_with_string_age():
    """Create a user dictionary with age as string"""
    return {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "age": "35"  # String that should be converted to int
    }


@pytest.fixture
def customer_dict():
    """Create a customer dictionary with addresses"""
    return {
        "name": "Customer A",
        "addresses": [
            {"street": "123 Main St", "city": "New York"},
            {"street": "456 Oak Ave", "city": "Los Angeles"}
        ]
    }


@pytest.fixture
def cyclic_dict():
    """Create dictionaries with cycle: A->B->C->A"""
    node_a = {"name": "Node A"}
    node_b = {"name": "Node B"}
    node_c = {"name": "Node C"}

    node_a["links_to"] = [node_b]
    node_b["links_to"] = [node_c]
    node_c["links_to"] = [node_a]  # Creates cycle

    return node_a


@pytest.fixture
def item_dicts():
    """Create multiple item dictionaries"""
    return [
        {"name": f"Item {i}", "quantity": i * 10}
        for i in range(1, 6)
    ]


# ===== Test Class =====

class TestDictConversion:
    """Tests for dictionary conversions"""

    def test_dict_to_ogm(self, db_connection, user_dict, user_dict_with_string_age):
        """
        Test converting a dictionary to an OGM model.

        Verifies that direct properties and simple relationships are correctly preserved.
        """
        # Convert to OGM
        user_ogm = Converter.dict_to_ogm(user_dict, UserOGM_DictTest)

        # Verify properties were preserved
        assert user_ogm.name == "John Doe", "Name not preserved"
        assert user_ogm.email == "john@example.com", "Email not preserved"
        assert user_ogm.age == 30, "Age not preserved"

        # Invalid type conversion will raise error and it's not the part of converter - to convert errones
        # to specified data types

    def test_ogm_to_dict(self, db_connection):
        """
        Test converting an OGM model to a dictionary.

        Verifies that properties are correctly extracted into a dictionary.
        """
        # Create an OGM instance
        product = ProductOGM_DictTest(name="Test Product", price=99.99, in_stock=True).save()

        # Convert to dictionary
        product_dict = Converter.ogm_to_dict(product)

        # Verify properties
        assert product_dict["name"] == "Test Product", "Name not preserved"
        assert product_dict["price"] == 99.99, "Price not preserved"
        assert product_dict["in_stock"] is True, "Stock status not preserved"

    def test_dict_relationship_conversion(self, db_connection, customer_dict):
        """
        Test converting dictionaries with nested relationships to OGM models.

        Verifies that relationships are properly established when converting from dictionaries.
        """
        # Convert to OGM
        customer_ogm = Converter.dict_to_ogm(customer_dict, CustomerOGM_DictTest)

        # Verify properties
        assert customer_ogm.name == "Customer A", "Name not preserved"

        # Verify relationships
        addresses = list(customer_ogm.addresses.all())
        assert len(addresses) == 2, "Not all addresses were created"

        # Verify relationship properties
        cities = sorted([a.city for a in addresses])
        assert cities == ["Los Angeles", "New York"], "Address cities not preserved"

        # Convert back to dictionary
        result_dict = Converter.ogm_to_dict(customer_ogm)

        # Verify dictionary structure
        assert result_dict["name"] == "Customer A", "Name not preserved in dict"
        assert len(result_dict["addresses"]) == 2, "Addresses not preserved in dict"
        assert "city" in result_dict["addresses"][0], "Address properties not preserved in dict"

    def test_dict_cyclic_references(self, db_connection, cyclic_dict):
        """
        Test handling of cyclic references in dictionaries.

        Verifies that circular references are properly handled without causing infinite recursion.
        """
        # Convert to OGM
        node_a_ogm = Converter.dict_to_ogm(cyclic_dict, NodeOGM_DictTest)

        # Verify properties
        assert node_a_ogm.name == "Node A", "Name not preserved"

        # Verify relationships (follow the cycle)
        nodes_b = list(node_a_ogm.links_to.all())
        assert len(nodes_b) == 1, "First link not created"
        assert nodes_b[0].name == "Node B", "First link name not preserved"

        nodes_c = list(nodes_b[0].links_to.all())
        assert len(nodes_c) == 1, "Second link not created"
        assert nodes_c[0].name == "Node C", "Second link name not preserved"

        # This should link back to A, completing the cycle
        nodes_a = list(nodes_c[0].links_to.all())
        assert len(nodes_a) == 1, "Third link not created"
        assert nodes_a[0].name == "Node A", "Third link name not preserved (cycle broken)"

        # Test OGM to dict with cycle
        result_dict = Converter.ogm_to_dict(node_a_ogm)

        # Verify the cycle is represented in the dict
        assert result_dict["name"] == "Node A", "Name not in dict"
        assert len(result_dict["links_to"]) == 1, "Links not in dict"
        assert result_dict["links_to"][0]["name"] == "Node B", "First link not in dict"
        assert result_dict["links_to"][0]["links_to"][0]["name"] == "Node C", "Second link not in dict"

        # The cycle should be detected and handled (either by empty dict or by reference to already processed node)
        assert "links_to" in result_dict["links_to"][0]["links_to"][0], "Cycle detection failed"


    def test_batch_dict_conversion(self, db_connection, item_dicts):
        """
        Test batch conversion of dictionaries to OGM models.

        Verifies that batch conversion works efficiently for multiple objects.
        """
        # Batch convert to OGM
        item_ogms = Converter.batch_dict_to_ogm(item_dicts, ItemOGM_DictTest)

        # Verify all items were converted
        assert len(item_ogms) == 5, "Not all items were converted"

        # Verify properties
        for i, item in enumerate(item_ogms):
            assert item.name == f"Item {i + 1}", f"Name not preserved for item {i + 1}"
            assert item.quantity == (i + 1) * 10, f"Quantity not preserved for item {i + 1}"

        # Batch convert back to dicts
        result_dicts = Converter.batch_ogm_to_dict(item_ogms)

        # Verify all dicts were created
        assert len(result_dicts) == 5, "Not all OGMs were converted to dicts"

        # Verify properties
        for i, item_dict in enumerate(result_dicts):
            assert item_dict["name"] == f"Item {i + 1}", f"Name not preserved in dict for item {i + 1}"
            assert item_dict["quantity"] == (i + 1) * 10, f"Quantity not preserved in dict for item {i + 1}"

    def test_dict_conversion_controls(self, db_connection):
        """
        Test controlling the dict conversion process with parameters.

        Verifies that include_properties and include_relationships flags work correctly.
        """
        # Define OGM models
        # Create an article with tags
        article = ArticleOGM_DictTest(title="Test Article", content="This is a test").save()
        tag1 = TagOGM_DictTest(name="Python").save()
        tag2 = TagOGM_DictTest(name="Neo4j").save()

        article.tags.connect(tag1)
        article.tags.connect(tag2)

        # Test with only properties
        props_only = Converter.ogm_to_dict(
            article,
            include_properties=True,
            include_relationships=False
        )

        # Verify only properties were included
        assert "title" in props_only, "Title property missing"
        assert "content" in props_only, "Content property missing"
        assert "tags" not in props_only, "Relationships should be excluded"

        # Test with only relationships
        rels_only = Converter.ogm_to_dict(
            article,
            include_properties=False,
            include_relationships=True
        )

        # Verify only relationships were included
        assert "title" not in rels_only, "Properties should be excluded"
        assert "content" not in rels_only, "Properties should be excluded"
        assert "tags" in rels_only, "Tags relationship missing"
        assert len(rels_only["tags"]) == 2, "Tags missing from relationships"

        # Test max_depth=0 parameter
        shallow = Converter.ogm_to_dict(article, max_depth=0)

        # Verify shallow conversion only includes properties
        assert "title" in shallow, "Title property missing in shallow conversion"
        assert "tags" not in shallow, "Relationships should be excluded in shallow conversion"

    def test_dict_to_ogm_max_depth(self, db_connection):
        """Test max_depth parameter in dict_to_ogm to ensure it stops at specified depth"""
        # Create a deeply nested dictionary structure
        level3 = {"name": "Level 3 Node", "links_to": []}
        level2 = {"name": "Level 2 Node", "links_to": [level3]}
        level1 = {"name": "Level 1 Node", "links_to": [level2]}
        root = {"name": "Root Node", "links_to": [level1]}

        # Convert with max_depth=2 (should convert root and level1, but not level2's relationships)
        result = Converter.dict_to_ogm(root, NodeOGM_DictTest, max_depth=2)

        # Verify the root and level1 were converted
        assert result is not None
        assert result.name == "Root Node"

        # Check level1 nodes
        level1_nodes = list(result.links_to.all())
        assert len(level1_nodes) == 1
        assert level1_nodes[0].name == "Level 1 Node"

        # Check level2 nodes (should exist as nodes, but without relationships)
        level2_nodes = list(level1_nodes[0].links_to.all())
        assert len(level2_nodes) == 1
        assert level2_nodes[0].name == "Level 2 Node"

        # Check level3 nodes - should not exist because level2's relationships weren't processed
        level3_nodes = list(level2_nodes[0].links_to.all())
        assert len(level3_nodes) == 0  # Nothing at level 3 due to max_depth=2
