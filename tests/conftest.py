import pytest
from neomodel import config, db


@pytest.fixture(scope="session", autouse=True)
def db_connection():
    """Setup Neo4j database connection for all tests"""
    config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'
    config.ENCRYPTED_CONNECTION = False
    config.AUTO_INSTALL_LABELS = True

    # Test connection first to fail early if there's an issue
    try:
        db.cypher_query("RETURN 1")
        print("Neo4j connection successful")
    except Exception as e:
        pytest.fail(f"Failed to connect to Neo4j: {str(e)}")

    yield
    # Clear database after all tests
    try:
        db.cypher_query("MATCH (n) DETACH DELETE n")
    except Exception as e:
        print(f"Warning: Could not clear database: {str(e)}")


@pytest.fixture(autouse=True)
def clear_database(db_connection):
    """Clear database before and after each test"""
    try:
        db.cypher_query("MATCH (n) DETACH DELETE n")
    except Exception as e:
        pytest.fail(f"Failed to clear database: {str(e)}")
    yield
    try:
        db.cypher_query("MATCH (n) DETACH DELETE n")
    except Exception as e:
        print(f"Warning: Could not clear database after test: {str(e)}")
