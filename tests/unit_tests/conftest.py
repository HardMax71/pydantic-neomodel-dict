import pytest

from pydantic_neo4j_dict import Converter


@pytest.fixture(autouse=True)
def clean_registry(request):
    """Reset the converter registry between unit tests to ensure test isolation"""
    # Only run this fixture for unit tests (path-based check)
    if "unit_tests" in request.path.parts:
        yield
        # Clean up the registry
        Converter._pydantic_to_ogm = {}
        Converter._ogm_to_pydantic = {}
        Converter._type_converters = {}
    else:
        # For hypothesis tests or any other test type, do nothing
        yield
