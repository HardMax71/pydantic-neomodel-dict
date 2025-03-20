import pytest
from neo4j import GraphDatabase
from neomodel import config, db

CYPHER_CLEAR_DB_QUERY = "MATCH (n) DETACH DELETE n"


@pytest.fixture(scope="session", autouse=True)
def db_connection():
    uri = 'bolt://localhost:7687'
    auth = ('neo4j', 'password')
    driver = GraphDatabase.driver(
        uri,
        auth=auth,
        encrypted=False,
        # Set minimum severity to WARNING, INFO will return tons of BS about cartesian product
        notifications_min_severity='WARNING',
        notifications_disabled_categories=['PERFORMANCE']
    )

    # Assign the custom driver to neomodel's config
    config.DRIVER = driver

    # Test connection first to fail early if there's an issue
    try:
        db.set_connection(driver=driver)
        db.cypher_query("RETURN 1")
        print("Neo4j connection successful")
    except Exception as e:
        pytest.fail(f"Failed to connect to Neo4j: {str(e)}")

    yield

    try:
        db.cypher_query(CYPHER_CLEAR_DB_QUERY)
    except Exception as e:
        print(f"Warning: Could not clear database: {str(e)}")

    driver.close()


@pytest.fixture(autouse=True, scope="function")
def clear_database(db_connection):
    """Clear database before and after each test."""
    try:
        db.cypher_query(CYPHER_CLEAR_DB_QUERY)
    except Exception as e:
        pytest.fail(f"Failed to clear database: {str(e)}")
    yield
    try:
        db.cypher_query(CYPHER_CLEAR_DB_QUERY)
    except Exception as e:
        print(f"Warning: Could not clear database after test: {str(e)}")
