import pytest
from pydantic import BaseModel

from pydantic_neomodel_dict import ConversionError, Converter

# ===== Module-level model definitions =====

class UnregisteredPydantic(BaseModel):
    name: str


class RequiredFieldPydantic(BaseModel):
    name: str  # Required field


# ===== Test Class =====

class TestErrorHandling:
    """Tests for error handling"""

    def test_unregistered_model(self, db_connection):
        """
        Test handling of unregistered model conversion attempts.

        Verifies that appropriate errors are raised when trying to convert
        models that haven't been registered with the converter.
        """
        # Create a Pydantic model without registration
        unregistered = UnregisteredPydantic(name="Test")

        # Try to convert - should raise an error
        with pytest.raises(ConversionError) as excinfo:
            Converter.to_ogm(unregistered)

        # Verify error contains meaningful message
        assert "No mapping registered for Pydantic class" in str(excinfo.value)

    def test_missing_required_field(self, db_connection):
        """
        Test handling of missing required fields.

        Verifies that appropriate validation errors are raised when required
        fields are missing from models.
        """
        # Try to create instance without required field
        with pytest.raises(Exception) as excinfo:
            RequiredFieldPydantic()

        # Verify error contains field name
        assert "name" in str(excinfo.value).lower()
