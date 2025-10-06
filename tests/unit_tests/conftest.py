import pytest

from pydantic_neomodel_dict.converters import SyncConverter

Converter = SyncConverter()


@pytest.fixture(autouse=True)
def clean_registry(request):
    """Reset the converter registry between unit tests to ensure test isolation"""
    # Only run this fixture for unit tests (path-based check)
    if "unit_tests" in request.path.parts:
        yield
        Converter.clear_registry()
    else:
        # For hypothesis tests or any other test type, do nothing
        yield
