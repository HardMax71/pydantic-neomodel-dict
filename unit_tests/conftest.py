import pytest
from neomodel import config, db
from converter import Converter

@pytest.fixture(scope="session")
def db_connection():
    """Setup Neo4j database connection for all tests"""
    config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'
    config.ENCRYPTED_CONNECTION = False
    config.AUTO_INSTALL_LABELS = True
    yield
    # Clear database after all tests
    db.cypher_query("MATCH (n) DETACH DELETE n")

@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the converter registry between tests to ensure test isolation"""
    yield
    # Original cleanup
    Converter._pydantic_to_ogm = {}
    Converter._ogm_to_pydantic = {}
    Converter._type_converters = {}

@pytest.fixture(autouse=True)
def clear_database():
    """Clear database before and after each unit test"""
    db.cypher_query("MATCH (n) DETACH DELETE n")
    yield
    db.cypher_query("MATCH (n) DETACH DELETE n")