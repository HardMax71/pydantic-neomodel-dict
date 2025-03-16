"""Pydantic ↔ Neo4j OGM ↔ Python Dict Converter.

A bidirectional converter between Pydantic models and Neo4j OGM (Object Graph Mapper) models,
supporting complex relationships, nested models, and custom type conversions.
"""

from .converter import ConversionError, Converter

__version__ = "0.1.0"
__all__ = ["Converter", "ConversionError"]
