from typing import Optional

import pytest
from neomodel import StructuredNode, StringProperty
from pydantic import BaseModel

from converter import Converter


# ===== Module-level model definitions =====

class ArticlePydantic(BaseModel):
    title: str
    author: Optional['AuthorPydantic'] = None


class AuthorPydantic(BaseModel):
    name: str
    bio: Optional[str] = None


# OGM model missing the relationship
class ArticleOGM(StructuredNode):
    title = StringProperty(required=True)
    # Missing author relationship


class AuthorOGM(StructuredNode):
    name = StringProperty(required=True)
    bio = StringProperty()


# ===== Fixtures =====

@pytest.fixture
def registered_models():
    """Register models"""
    Converter.register_models(ArticlePydantic, ArticleOGM)
    Converter.register_models(AuthorPydantic, AuthorOGM)
    yield


@pytest.fixture
def article_with_author():
    """Create an article with an author"""
    author = AuthorPydantic(name="Jane Doe", bio="Technical writer")
    return ArticlePydantic(title="Understanding Graph Databases", author=author)


# ===== Test Class =====

class TestMissingRelationships:
    """Tests for handling missing relationships"""

    def test_missing_relationship_on_ogm(self, db_connection, registered_models, article_with_author):
        """
        Test handling of missing relationships on OGM side.

        Verifies that conversion still succeeds when a relationship exists in the
        Pydantic model but not in the corresponding OGM model.
        """
        # Get article from fixture
        article = article_with_author

        # Convert to OGM - should succeed despite missing relationship
        article_ogm = Converter.to_ogm(article)

        # Verify properties
        assert article_ogm.title == "Understanding Graph Databases", "Title not preserved"

        # Convert back to Pydantic
        converted_back = Converter.to_pydantic(article_ogm)

        # Verify properties (author should be None)
        assert converted_back.title == article.title, "Title not preserved in round trip"
        assert converted_back.author is None, "Missing relationship should result in None"
