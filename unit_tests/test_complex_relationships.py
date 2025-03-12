from typing import List

import pytest
from neomodel import (
    StructuredNode, StringProperty, DateTimeProperty,
    RelationshipTo, StructuredRel
)
from pydantic import BaseModel, Field

from converter import Converter


# ===== Module-level model definitions =====

# Relationship model
class AuthoredRel(StructuredRel):
    created_at = DateTimeProperty()


# Pydantic models
class CommentPydantic(BaseModel):
    content: str
    author: str


class PostPydantic(BaseModel):
    title: str
    content: str
    comments: List[CommentPydantic] = Field(default_factory=list)


class BloggerPydantic(BaseModel):
    name: str
    bio: str
    posts: List[PostPydantic] = Field(default_factory=list)
    following: List['BloggerPydantic'] = Field(default_factory=list)


# Resolve forward references
BloggerPydantic.model_rebuild()


# Neo4j OGM models
class CommentOGM(StructuredNode):
    content = StringProperty(required=True)
    author = StringProperty(required=True)


class PostOGM(StructuredNode):
    title = StringProperty(required=True)
    content = StringProperty(required=True)
    comments = RelationshipTo(CommentOGM, 'HAS_COMMENT')


class BloggerOGM(StructuredNode):
    name = StringProperty(required=True)
    bio = StringProperty()
    posts = RelationshipTo(PostOGM, 'AUTHORED', model=AuthoredRel)
    following = RelationshipTo('BloggerOGM', 'FOLLOWS')


# ===== Fixtures =====

@pytest.fixture
def registered_models():
    """Register models"""
    Converter.register_models(CommentPydantic, CommentOGM)
    Converter.register_models(PostPydantic, PostOGM)
    Converter.register_models(BloggerPydantic, BloggerOGM)
    yield


@pytest.fixture
def blogger_fixture():
    """Create test blogger with posts and comments"""
    # Create test data
    comment1 = CommentPydantic(content="Great post!", author="Reader1")
    comment2 = CommentPydantic(content="I learned a lot", author="Reader2")

    post1 = PostPydantic(
        title="Introduction to Neo4j",
        content="Neo4j is a graph database...",
        comments=[comment1, comment2]
    )

    post2 = PostPydantic(
        title="Pydantic and Data Validation",
        content="Pydantic provides data validation...",
        comments=[CommentPydantic(content="Very helpful", author="Reader3")]
    )

    blogger = BloggerPydantic(
        name="Tech Writer",
        bio="I write about tech",
        posts=[post1, post2]
    )

    # Add self-reference
    another_blogger = BloggerPydantic(
        name="Code Expert",
        bio="Professional developer",
        posts=[]
    )
    blogger.following = [another_blogger]

    return blogger


# ===== Test Class =====

class TestComplexRelationships:
    """Tests for models with complex nested relationships"""

    def test_nested_relationships(self, db_connection, registered_models, blogger_fixture):
        """
        Test converting models with complex nested relationships (lists and depth).

        Verifies that multi-level relationships (blogger->posts->comments) and self-references
        are properly handled during conversion.
        """
        # Get blogger from fixture
        blogger = blogger_fixture

        # Convert to OGM
        blogger_ogm = Converter.to_ogm(blogger)

        # Verify blogger properties
        assert blogger_ogm.name == "Tech Writer", "Blogger name not preserved"
        assert blogger_ogm.bio == "I write about tech", "Blogger bio not preserved"

        # Verify posts relationship
        posts = list(blogger_ogm.posts.all())
        assert len(posts) == 2, "Incorrect number of posts"

        # Find the post with specific title
        pyd_post = next((p for p in posts if p.title == "Pydantic and Data Validation"), None)
        assert pyd_post is not None, "Post 'Pydantic and Data Validation' not found"

        # Verify comments
        comments = list(pyd_post.comments.all())
        assert len(comments) == 1, "Incorrect number of comments"
        assert comments[0].content == "Very helpful", "Comment content not preserved"
        assert comments[0].author == "Reader3", "Comment author not preserved"

        # Convert back to Pydantic
        converted_back = Converter.to_pydantic(blogger_ogm)

        # Verify structure after round trip
        assert converted_back.name == blogger.name, "Blogger name not preserved in round trip"
        assert len(converted_back.posts) == 2, "Incorrect number of posts in round trip"

        # Find post in converted data
        converted_post = next((p for p in converted_back.posts if p.title == "Pydantic and Data Validation"), None)
        assert converted_post is not None, "Post not found in round trip"
        assert len(converted_post.comments) == 1, "Incorrect number of comments in round trip"
        assert converted_post.comments[0].content == "Very helpful", "Comment content not preserved in round trip"
