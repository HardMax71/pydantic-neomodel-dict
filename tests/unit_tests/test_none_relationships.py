from typing import Optional

import pytest
from neomodel import RelationshipTo, StringProperty, StructuredNode, ZeroOrOne
from pydantic import BaseModel

from pydantic_neomodel_dict.converters import SyncConverter

Converter = SyncConverter()

# ===== Module-level model definitions =====

class ProductPydantic(BaseModel):
    name: str
    category: Optional['CategoryPydantic'] = None


class CategoryPydantic(BaseModel):
    name: str
    description: Optional[str] = None


class ProductOGM(StructuredNode):
    name = StringProperty(required=True)
    category = RelationshipTo('CategoryOGM', 'BELONGS_TO', cardinality=ZeroOrOne)


class CategoryOGM(StructuredNode):
    name = StringProperty(required=True)
    description = StringProperty()


# Resolve forward reference
ProductPydantic.model_rebuild()


# ===== Fixtures =====

@pytest.fixture
def registered_models():
    """Register models"""
    Converter.register_models(ProductPydantic, ProductOGM)
    Converter.register_models(CategoryPydantic, CategoryOGM)
    yield


@pytest.fixture
def product_with_none():
    """Create a product with None category"""
    return ProductPydantic(name="Test Product", category=None)


@pytest.fixture
def product_with_category():
    """Create a product with a category"""
    category = CategoryPydantic(name="Electronics", description="Electronic products")
    return ProductPydantic(name="Smartphone", category=category)


# ===== Test Class =====

class TestNoneRelationships:
    """Tests for handling None values in relationships"""

    def test_none_relationships(self, db_connection, registered_models, product_with_none, product_with_category):
        """
        Test handling of None values in relationships.

        Verifies that None relationships are properly handled in both
        Pydantic to OGM and OGM to Pydantic conversions.
        """
        # Test 1: Product with None category
        product_with_none_fixture = product_with_none

        # Convert to OGM
        product_ogm = Converter.to_ogm(product_with_none_fixture)

        # Verify properties
        assert product_ogm.name == "Test Product", "Product name not preserved"
        assert len(list(product_ogm.category.all())) == 0, "None category should result in no relationships"

        # Convert back to Pydantic
        converted_back = Converter.to_pydantic(product_ogm)

        # Verify properties
        assert converted_back.name == product_with_none_fixture.name, "Product name not preserved in round trip"
        assert converted_back.category is None, "None category not preserved in round trip"

        # Test 2: Product with category
        product_with_category_fixture = product_with_category

        # Convert to OGM
        product_with_cat_ogm = Converter.to_ogm(product_with_category_fixture)

        # Verify relationship created
        categories = list(product_with_cat_ogm.category.all())
        assert len(categories) == 1, "Category relationship not created"
        assert categories[0].name == "Electronics", "Category name not preserved"

        # Convert back to Pydantic
        converted_with_cat = Converter.to_pydantic(product_with_cat_ogm)

        # Verify relationship preserved
        assert converted_with_cat.category is not None, "Category relationship lost in round trip"
        assert converted_with_cat.category.name == "Electronics", "Category name not preserved in round trip"
