import pytest
from neomodel import config, db
from hypothesis import settings, Verbosity, HealthCheck
import os

# Configure Hypothesis profiles
settings.register_profile(
    "ci",
    max_examples=10,  # Significantly reduce test examples in CI
    deadline=None,    # Disable deadline checking in CI
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large
    ],
    verbosity=Verbosity.verbose
)

settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow]
)

# Load CI profile when running in GitHub Actions
if os.environ.get("GITHUB_ACTIONS") == "true":
    settings.load_profile("ci")
else:
    settings.load_profile("dev")

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
def clear_database():
    """Clear database before and after each hypothesis test"""
    db.cypher_query("MATCH (n) DETACH DELETE n")
    yield
    db.cypher_query("MATCH (n) DETACH DELETE n")

# We're removing the clean_registry fixture to preserve registrations
# between tests in the hypothesis test file