"""
Pydantic to Neo4j OGM Converter.

This module provides a utility for converting between Pydantic models and Neo4j OGM models
with support for relationships, nested models, and custom type conversions.
"""
import inspect
import logging
from datetime import datetime, date
from functools import lru_cache
from typing import Type, Dict, List, Any, Optional, Union, Tuple, get_type_hints, Callable, Set, Generic, TypeVar

from neomodel import (
    StructuredNode, RelationshipTo, RelationshipFrom,
    Relationship, db
)
from neomodel import ZeroOrOne, One, AsyncZeroOrOne, AsyncOne
from neomodel.properties import (
    StringProperty, IntegerProperty, FloatProperty, BooleanProperty,
    DateTimeProperty, ArrayProperty, JSONProperty
)
from pydantic import BaseModel

# Type variables for generic typing
PydanticModel = TypeVar('PydanticModel', bound=BaseModel)
OGM_Model = TypeVar('OGM_Model', bound=StructuredNode)

# Configure logger
logger = logging.getLogger(__name__)


class ConversionError(Exception):
    """Exception raised for errors during model conversion."""
    pass


class Converter(Generic[PydanticModel, OGM_Model]):
    """
    A utility class for converting between Pydantic models and neomodel OGM models.

    This converter handles:
    - Basic property conversion between models.
    - Conversion of nested Pydantic models into related OGM nodes.
    - Processing of relationships at any depth (including cyclic references).
    - Conversion of lists and dictionaries of models.
    - Custom type conversions via user-registered converters.
    - Conversion from Python dictionaries to OGM models (via dict_to_ogm)
      and from OGM models to Python dictionaries (via ogm_to_dict).
    - Batch conversion of multiple models (via batch_to_ogm and batch_to_pydantic).
    """

    # Registry to store mappings between Pydantic and OGM models
    _pydantic_to_ogm: Dict[Type[BaseModel], Type[StructuredNode]] = {}
    _ogm_to_pydantic: Dict[Type[StructuredNode], Type[BaseModel]] = {}

    # Custom type converters
    _type_converters: Dict[Tuple[Type, Type], Callable[[Any], Any]] = {}

    @classmethod
    def register_type_converter(
            cls,
            source_type: Type,
            target_type: Type,
            converter_func: Callable[[Any], Any]
    ) -> None:
        """
        Register a custom type converter function.

        Args:
            source_type (Type): The source type to convert from.
            target_type (Type): The target type to convert to.
            converter_func (Callable[[Any], Any]): A function that converts a value from source_type to target_type.

        Returns:
            None.
        """
        cls._type_converters[(source_type, target_type)] = converter_func
        logger.debug(f"Registered type converter: {source_type.__name__} -> {target_type.__name__}")

    @classmethod
    @lru_cache(maxsize=128)
    def _is_property(cls, attr: Any) -> bool:
        """
        Determine if the given attribute is a neomodel property.

        This function checks whether the attribute is an instance of any known neomodel property type.

        Args:
            attr (Any): The attribute to check.

        Returns:
            bool: True if the attribute is a neomodel property, False otherwise.
        """
        try:
            property_types = (
                StringProperty, IntegerProperty, FloatProperty, BooleanProperty,
                DateTimeProperty, ArrayProperty, JSONProperty
            )
            return isinstance(attr, property_types)
        except TypeError:
            return False

    @classmethod
    @lru_cache(maxsize=128)
    def _is_relationship(cls, attr: Any) -> bool:
        """
        Determine if the given attribute is a neomodel relationship.

        Args:
            attr (Any): The attribute to check.

        Returns:
            bool: True if the attribute is a neomodel relationship, False otherwise.
        """
        return isinstance(attr, (RelationshipTo, RelationshipFrom, Relationship))

    @classmethod
    def _get_ogm_properties(cls, ogm_class: Type[StructuredNode]) -> Dict[str, Any]:
        """
        Retrieve all property fields of a neomodel StructuredNode class.

        Args:
            ogm_class (Type[StructuredNode]): A neomodel StructuredNode class.

        Returns:
            Dict[str, Any]: A dictionary mapping property names to their corresponding property instances.
        """
        # Use __all_properties__ attribute which is available in StructuredNode classes
        if hasattr(ogm_class, '__all_properties__'):
            return dict(ogm_class.__all_properties__)

        # As a fallback, use defined_properties method if available
        if hasattr(ogm_class, 'defined_properties'):
            return ogm_class.defined_properties(aliases=False, rels=False)

        # Manual inspection as last resort
        return {
            name: attr
            for name, attr in inspect.getmembers(ogm_class)
            if not name.startswith('_') and cls._is_property(attr)
        }

    @classmethod
    def _get_ogm_relationships(
            cls, ogm_class: Type[StructuredNode]
    ) -> Dict[str, Union[RelationshipTo, RelationshipFrom, Relationship]]:
        """
        Retrieve all relationship fields of a neomodel StructuredNode class.

        Args:
            ogm_class (Type[StructuredNode]): A neomodel StructuredNode class.

        Returns:
            Dict[str, Union[RelationshipTo, RelationshipFrom, Relationship]]:
                A dictionary mapping relationship names to their corresponding relationship instances.
        """
        # Use __all_relationships__ attribute which is available in StructuredNode classes
        if hasattr(ogm_class, '__all_relationships__'):
            return dict(ogm_class.__all_relationships__)

        # As a fallback, use defined_properties method if available
        if hasattr(ogm_class, 'defined_properties'):
            return ogm_class.defined_properties(aliases=False, rels=True, properties=False)

        # Manual inspection as last resort
        return {
            name: attr
            for name, attr in inspect.getmembers(ogm_class)
            if not name.startswith('_') and cls._is_relationship(attr)
        }

    @classmethod
    def _get_property_type(cls, prop: Any) -> Any:  # Return Any instead of Type to fix error
        """
        Determine the Python type corresponding to a neomodel property.

        Args:
            prop (Any): A neomodel property.

        Returns:
            Type: The Python type associated with the property.
        """
        # Map neomodel property types to Python types
        if isinstance(prop, StringProperty):
            return str
        elif isinstance(prop, IntegerProperty):
            return int
        elif isinstance(prop, FloatProperty):
            return float
        elif isinstance(prop, BooleanProperty):
            return bool
        elif isinstance(prop, DateTimeProperty):
            return datetime
        elif isinstance(prop, ArrayProperty):
            return list
        elif isinstance(prop, JSONProperty):
            return dict

        # For any other property types, try to get the type from the property_type attribute
        # or use Any as a fallback
        try:
            return getattr(prop, 'property_type', Any)
        except (AttributeError, TypeError):
            return Any

    @classmethod
    def _convert_value(cls, value: Any, target_type: Any) -> Any:
        """
        Convert the given value to the specified target type using registered converters if available.

        Args:
            value (Any): The value to convert.
            target_type (Type): The target type to which the value should be converted.

        Returns:
            Any: The converted value.
        """
        if value is None:
            return None

        source_type = type(value)

        # Check for direct registered converter
        converter = cls._type_converters.get((source_type, target_type))
        if converter:
            return converter(value)

        # Handle basic type conversions
        if target_type is str and not isinstance(value, str):
            return str(value)

        if target_type is int and isinstance(value, (str, float)):
            return int(value)

        if target_type is float and isinstance(value, (str, int)):
            return float(value)

        if target_type is bool and not isinstance(value, bool):
            return bool(value)

        if target_type is datetime and isinstance(value, str):
            try:
                # Try ISO format first
                return datetime.fromisoformat(value)
            except ValueError:
                # Try other common formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue

                # If all else fails, raise an error
                raise ConversionError(f"Cannot convert string '{value}' to datetime")

        if target_type is date and isinstance(value, str):
            try:
                # Try ISO format first
                return date.fromisoformat(value)
            except ValueError:
                # Try other common formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]:
                    try:
                        return datetime.strptime(value, fmt).date()
                    except ValueError:
                        continue

                # If all else fails, raise an error
                raise ConversionError(f"Cannot convert string '{value}' to date")

        # If the target_type is a class and the value is a dict, attempt to create an instance
        if isinstance(target_type, type) and issubclass(target_type, BaseModel) and isinstance(value, dict):
            return target_type(**value)

        # If we get here, just return the original value
        return value

    @classmethod
    def register_models(cls, pydantic_class: Type[BaseModel], ogm_class: Type[StructuredNode]) -> None:
        """
        Register a mapping between a Pydantic model class and a neomodel OGM model class.

        Args:
            pydantic_class (Type[BaseModel]): The Pydantic model class
            ogm_class (Type[StructuredNode]): The neomodel OGM model class
        """
        cls._pydantic_to_ogm[pydantic_class] = ogm_class
        cls._ogm_to_pydantic[ogm_class] = pydantic_class
        logger.debug(f"Registered mapping: {pydantic_class.__name__} <-> {ogm_class.__name__}")

    @classmethod
    def _process_pydantic_field(cls, pydantic_instance: BaseModel, field_name: str,
                                pydantic_data: Dict[str, Any]) -> None:
        """
        Process a single field from a Pydantic model, handling BaseModel instances and lists of BaseModels.

        This helper function extracts a field value from a Pydantic model instance,
        performs special handling for BaseModel instances and lists of BaseModels
        to avoid circular references, and updates the provided data dictionary.

        Args:
            pydantic_instance: The Pydantic model instance
            field_name: The name of the field to process
            pydantic_data: The dictionary to update with the processed field value

        Returns:
            None - updates pydantic_data in-place
        """
        sentinel = object()
        value = getattr(pydantic_instance, field_name, sentinel)
        if value is sentinel or isinstance(value, BaseModel):
            return
        if isinstance(value, list) and all(isinstance(item, BaseModel) for item in value):
            seen_ids = set()
            processed_list = []
            for item in value:
                if id(item) not in seen_ids:
                    seen_ids.add(id(item))
                    processed_list.append(item)
            value = processed_list
        pydantic_data[field_name] = value

    @classmethod
    def _resolve_node_class(cls, class_name_or_obj: Union[str, Type[StructuredNode]]) -> Optional[Type[StructuredNode]]:
        """
        Resolve a class name or object to an actual class using a direct lookup in the registered mappings.

        Args:
            class_name_or_obj: Either a string class name or a class object.

        Returns:
            The resolved class or None if not found.
        """
        # If already a class, just return it.
        if not isinstance(class_name_or_obj, str):
            return class_name_or_obj

        # Build a simple mapping from class name to the OGM class based on our registered models.
        mapping = {ogm_cls.__name__: ogm_cls for ogm_cls in cls._ogm_to_pydantic.keys()}
        if class_name_or_obj in mapping:
            return mapping[class_name_or_obj]

        # Fall back to searching in Neo4j's registry.
        for label_set, cls_obj in db._NODE_CLASS_REGISTRY.items():
            if cls_obj.__name__ == class_name_or_obj:
                return cls_obj

        return None

    @classmethod
    def to_ogm(
            cls,
            pydantic_instance: BaseModel,  # Change from PydanticModel to BaseModel
            ogm_class: Optional[Type[OGM_Model]] = None,
            processed_objects: Optional[Dict[int, OGM_Model]] = None,
            max_depth: int = 10
    ) -> Optional[OGM_Model]:
        """
        Convert a Pydantic model instance to a neomodel OGM model instance.

        Args:
            pydantic_instance (BaseModel): The Pydantic model instance to convert.
            ogm_class (Optional[Type[OGM_Model]]): The target neomodel OGM model class. If not provided,
                the registered mapping is used.
            processed_objects (Optional[Dict[int, OGM_Model]]): A dictionary to track already processed objects
                for handling cyclic references.
            max_depth (int): The maximum recursion depth for processing nested relationships.

        Returns:
            Optional[OGM_Model]: The converted neomodel OGM model instance.
        """
        if pydantic_instance is None:
            return None

        if max_depth <= 0:
            logger.info(f"Maximum recursion depth reached for {type(pydantic_instance).__name__}")
            return None

        # Do not decrement max_depth for the root object itself.
        if processed_objects is None:
            processed_objects = {}

        instance_id = id(pydantic_instance)
        if instance_id in processed_objects:
            return processed_objects[instance_id]

        if ogm_class is None:
            pydantic_class = type(pydantic_instance)
            if pydantic_class not in cls._pydantic_to_ogm:
                raise ConversionError(f"No mapping registered for Pydantic class {pydantic_class.__name__}")
            ogm_class = cls._pydantic_to_ogm[pydantic_class]

        # Create the OGM instance.
        ogm_instance = ogm_class()
        processed_objects[instance_id] = ogm_instance

        # Extract Pydantic data.
        pydantic_data: Dict[str, Any] = {}
        try:
            pydantic_data = pydantic_instance.model_dump(exclude_unset=True, exclude_none=True)
        except ValueError as e:
            if "Circular reference detected" in str(e):
                for field_name in pydantic_instance.model_fields.keys():
                    cls._process_pydantic_field(pydantic_instance, field_name, pydantic_data)
            else:
                raise ConversionError(f"Failed to get dictionary from Pydantic model: {str(e)}")

        ogm_properties = cls._get_ogm_properties(ogm_class)
        for prop_name, prop in ogm_properties.items():
            if prop_name in pydantic_data:
                value = pydantic_data[prop_name]
                # Any neomodel OGM field is successor of Property class, implementing `deflate()` method,
                # That returns value of field as python type.
                # TODO: maybe to replace with: setattr(ogm_instance, prop_name, prop.deflate(value))
                prop_type = cls._get_property_type(prop)
                setattr(ogm_instance, prop_name, cls._convert_value(value, prop_type))

        # Save the object with properties before processing relationships.
        ogm_instance.save()

        # Process relationships if we have depth remaining.
        sentinel = object()
        ogm_relationships = cls._get_ogm_relationships(ogm_class)
        for rel_name, rel in ogm_relationships.items():
            # Skip inverse relationships since they are automatically managed.
            # if isinstance(rel, RelationshipFrom):
            #    continue

            rel_data = getattr(pydantic_instance, rel_name, sentinel)
            if rel_data is sentinel:
                logger.warning(f"Failed to get relationship {rel_name}")
                continue
            if rel_data is None:
                continue

            # Resolve target class.
            target_ogm_class = cls._resolve_node_class(rel.definition['node_class'])
            if not target_ogm_class:
                logger.warning(f"Could not resolve target class for relationship {rel_name}")
                continue

            # Normalize to list if needed.
            if not isinstance(rel_data, list):
                rel_data = [rel_data]

            new_max_depth = max_depth - 1  # Decrement depth only once.
            for item in rel_data:
                cls._process_related_item(
                    item,
                    ogm_instance,
                    rel_name,
                    target_ogm_class,
                    processed_objects,
                    new_max_depth,  # Pass new_max_depth directly.
                    instance_id
                )

        # Save the complete object after processing relationships.
        ogm_instance.save()
        return ogm_instance

    @classmethod
    def _process_related_item(
            cls,
            item: Union[BaseModel, Dict[str, Any]],
            ogm_instance: OGM_Model,
            rel_name: str,
            target_ogm_class: Type[StructuredNode],
            processed_objects: Dict[int, OGM_Model],
            max_depth: int,
            instance_id: int
    ) -> bool:
        """
        Process a single related item and connect it to the OGM instance if successful.

        Args:
            item: The item to process (BaseModel or dict)
            ogm_instance: The OGM instance to connect the related item to
            rel_name: The name of the relationship
            target_ogm_class: The target OGM class
            processed_objects: Dictionary of already processed objects
            max_depth: Maximum recursion depth
            instance_id: ID of the parent instance (to avoid circular references)

        Returns:
            bool: Whether the item was successfully processed and connected
        """
        rel_manager = getattr(ogm_instance, rel_name)
        if isinstance(item, BaseModel):
            # Handle self–references: if the item is the same as the parent, simply connect.
            if id(item) == instance_id:
                rel_manager.connect(ogm_instance)
                return True

            related_instance = cls.to_ogm(
                item,
                target_ogm_class,
                processed_objects,
                max_depth
            )
            if related_instance:
                if isinstance(rel_manager, (ZeroOrOne, One, AsyncZeroOrOne, AsyncOne)):
                    # Use replace to remove any existing node and connect the new one.
                    rel_manager.replace(related_instance)
                else:
                    rel_manager.connect(related_instance)
                return True
        elif isinstance(item, dict):
            target_pydantic_class = cls._ogm_to_pydantic.get(target_ogm_class)
            if target_pydantic_class and isinstance(item, dict):
                related_pydantic = target_pydantic_class(**item)
                related_instance = cls.to_ogm(
                    related_pydantic,
                    target_ogm_class,
                    processed_objects,
                    max_depth  # Use the caller's decremented depth.
                )
                if related_instance:
                    if isinstance(rel_manager, (ZeroOrOne, One, AsyncZeroOrOne, AsyncOne)):
                        # Use replace to remove any existing node and connect the new one.
                        rel_manager.replace(related_instance)
                    else:
                        rel_manager.connect(related_instance)
                    return True

                raise ConversionError(f"Failed to process dict item in relationship {rel_name}")
            else:
                raise ConversionError(f"No Pydantic model registered for OGM class {target_ogm_class.__name__}")
        return False

    @classmethod
    def _create_minimal_pydantic_instance(
            cls,
            ogm_instance: OGM_Model,
            pydantic_class: Optional[Type[BaseModel]] = None
    ) -> BaseModel:
        """
        Create a minimal Pydantic instance with only essential properties.
        Used for cycle breaking and max depth handling.
        """
        # Resolve Pydantic class if not provided
        if pydantic_class is None:
            ogm_class = type(ogm_instance)
            if ogm_class not in cls._ogm_to_pydantic:
                raise ConversionError(f"No mapping registered for OGM class {ogm_class.__name__}")
            pydantic_class = cls._ogm_to_pydantic[ogm_class]

        # Extract essential properties
        sentinel = object()
        ogm_properties = cls._get_ogm_properties(type(ogm_instance))
        pydantic_data: Dict[str, Any] = {}

        for prop_name, prop in ogm_properties.items():
            # Prioritize required and unique index properties
            is_key_property = (hasattr(prop, 'required') and prop.required) or \
                              (hasattr(prop, 'unique_index') and prop.unique_index)

            if is_key_property or not pydantic_data:  # Include at least something if no keys found
                value = getattr(ogm_instance, prop_name, sentinel)
                if value is not sentinel:
                    pydantic_data[prop_name] = value

        return pydantic_class(**pydantic_data)

    @classmethod
    def _get_property_data(
            cls,
            ogm_instance: OGM_Model,
            pydantic_fields: dict
    ) -> dict:
        """Extract property data from OGM instance for Pydantic model creation."""
        sentinel = object()
        ogm_properties = cls._get_ogm_properties(type(ogm_instance))

        return {
            prop_name: cls._convert_value(value, pydantic_fields.get(prop_name, Any))
            for prop_name, prop in ogm_properties.items()
            if prop_name in pydantic_fields
            if (value := getattr(ogm_instance, prop_name, sentinel)) is not sentinel
        }

    @classmethod
    def _is_list_type(cls, field_type: Any) -> bool:
        """
        Determine if a field type is a List type using proper typing inspection.

        Args:
            field_type: The field type annotation

        Returns:
            True if the field is a list type, False otherwise
        """
        # Check for standard list type
        if field_type is list:
            return True

        # Handle typing.List
        try:
            import typing
            if hasattr(typing, 'get_origin') and hasattr(typing, 'List'):  # Python 3.8+
                origin = typing.get_origin(field_type)
                return origin is list or origin is typing.List
            # For Python 3.7 and below
            elif hasattr(field_type, "__origin__"):
                return field_type.__origin__ is list
        except (ImportError, AttributeError):
            pass

        # Fallback: check string representation (less ideal but works as last resort)
        field_str = str(field_type)
        return field_str.startswith('typing.List') or field_str.startswith('List[')

    @classmethod
    def batch_to_pydantic(
            cls,
            ogm_instances: List[OGM_Model],
            pydantic_class: Optional[Type[BaseModel]] = None,
            max_depth: int = 10
    ) -> List[BaseModel]:
        """
        Convert a list of neomodel OGM model instances to Pydantic model instances.

        Args:
            ogm_instances (List[OGM_Model]): A list of neomodel OGM model instances to convert.
            pydantic_class (Optional[Type[BaseModel]]): The target Pydantic model class.
                If not provided, the registered mapping is used.
            max_depth (int): The maximum recursion depth for processing nested relationships.

        Returns:
            List[BaseModel]: A list of converted Pydantic model instances.
        """
        if not ogm_instances:
            return []

        # Use a single processed_objects dictionary for the entire batch
        processed_objects: Dict[int, BaseModel] = {}

        result = []
        for instance in ogm_instances:
            pydantic_instance = cls.to_pydantic(instance, pydantic_class, processed_objects, max_depth, set())
            if pydantic_instance is not None:
                result.append(pydantic_instance)
        return result

    @classmethod
    def batch_to_ogm(
            cls,
            pydantic_instances: List[BaseModel],
            ogm_class: Optional[Type[OGM_Model]] = None,
            max_depth: int = 10
    ) -> List[OGM_Model]:
        """
        Convert a list of Pydantic model instances to neomodel OGM model instances within a single transaction.

        This method is optimized for batch conversion of multiple instances, utilizing a single database transaction
        for improved performance.

        Args:
            pydantic_instances (List[BaseModel]): A list of Pydantic model instances to convert.
            ogm_class (Optional[Type[OGM_Model]]): The target neomodel OGM model class.
                If not provided, the registered mapping is used.
            max_depth (int): The maximum recursion depth for processing nested relationships.

        Returns:
            List[OGM_Model]: A list of converted neomodel OGM model instances.

        Raises:
            ConversionError: If the conversion fails.
        """
        if not pydantic_instances:
            return []

        # Use a single processed_objects dictionary for the entire batch
        processed_objects: Dict[int, OGM_Model] = {}

        # Use a transaction for the entire batch
        result: List[OGM_Model] = []
        with db.transaction:
            for instance in pydantic_instances:
                ogm_instance = cls.to_ogm(instance, ogm_class, processed_objects, max_depth)
                if ogm_instance is not None:
                    result.append(ogm_instance)
        return result

    @classmethod
    def _dict_to_ogm_process_relationships(
            cls,
            ogm_instance: OGM_Model,
            data_dict: dict,
            ogm_relationships: Dict[str, Any],
            processed_objects: Dict[int, OGM_Model],
            max_depth: int
    ) -> None:
        """
        Process relationships for dict_to_ogm method.

        This method extracts relationship data from a dictionary and connects it to an OGM instance.
        It handles both dictionary and Pydantic model relationships.

        Args:
            ogm_instance: The OGM instance to connect relationships to
            data_dict: Source dictionary containing relationship data
            ogm_relationships: Dictionary of OGM relationship definitions
            processed_objects: Dictionary tracking already processed objects
            max_depth: Maximum recursion depth for nested relationships
        """
        for rel_name, rel in ogm_relationships.items():
            if rel_name not in data_dict or data_dict[rel_name] is None:
                continue

            rel_data = data_dict[rel_name]
            target_ogm_class = cls._resolve_node_class(rel.definition['node_class'])
            if not target_ogm_class:
                raise ConversionError(f"Could not resolve target class for relationship {rel_name}")

            rel_manager = getattr(ogm_instance, rel_name)

            # Normalize to list if needed
            if not isinstance(rel_data, list):
                rel_data = [rel_data]

            new_max_depth = max_depth - 1
            for item in rel_data:
                if isinstance(item, BaseModel):
                    # Handle Pydantic models directly
                    related_instance = cls.to_ogm(item, target_ogm_class, processed_objects, new_max_depth)
                    if related_instance:
                        cls._connect_related_instance(rel_manager, related_instance)
                elif isinstance(item, dict):
                    # Handle dictionaries
                    related_instance = cls.dict_to_ogm(item, target_ogm_class, processed_objects, new_max_depth)
                    if related_instance:
                        cls._connect_related_instance(rel_manager, related_instance)

    @classmethod
    def _connect_related_instance(cls, rel_manager: Any, related_instance: OGM_Model) -> None:
        """Helper method to connect a related instance to a relationship manager"""
        if isinstance(rel_manager, (ZeroOrOne, One, AsyncZeroOrOne, AsyncOne)):
            try:
                # Disconnect all previously connected nodes for one-to-one relationships
                rel_manager.disconnect_all()
            except Exception as e:
                logger.debug(f"Could not disconnect existing relationships: {e}")
            rel_manager.connect(related_instance)
        else:
            rel_manager.connect(related_instance)

    @classmethod
    def dict_to_ogm(
            cls,
            data_dict: dict,
            ogm_class: Type[OGM_Model],
            processed_objects: Optional[Dict[int, OGM_Model]] = None,
            max_depth: int = 10
    ) -> Optional[OGM_Model]:
        """
        Convert a Python dictionary to a neomodel OGM model instance.

        This function recursively converts a dictionary (including nested dictionaries)
        into a neomodel OGM model instance, handling relationships and nested objects.

        Args:
            data_dict: Source dictionary containing data to convert
            ogm_class: Target OGM class for conversion
            processed_objects: Dictionary tracking already processed objects to handle cycles
            max_depth: Maximum recursion depth for nested relationships

        Returns:
            A new or updated OGM model instance, or None if input is None
        """
        if data_dict is None:
            return None

        if max_depth <= 0:
            logger.warning("Maximum recursion depth reached during dict_to_ogm conversion")
            return None

        processed_objects = processed_objects or {}

        # Use id(data_dict) for cycle detection
        instance_id = id(data_dict)
        if instance_id in processed_objects:
            return processed_objects[instance_id]

        # Create a new OGM instance and register it
        ogm_instance = ogm_class()
        processed_objects[instance_id] = ogm_instance

        # Process properties
        for prop_name, prop in cls._get_ogm_properties(ogm_class).items():
            if prop_name in data_dict:
                value = data_dict[prop_name]
                prop_type = cls._get_property_type(prop)
                setattr(ogm_instance, prop_name, cls._convert_value(value, prop_type))

        # Save properties before processing relationships
        ogm_instance.save()

        # Process relationships
        ogm_relationships = cls._get_ogm_relationships(ogm_class)
        cls._dict_to_ogm_process_relationships(
            ogm_instance, data_dict, ogm_relationships, processed_objects, max_depth
        )

        # Final save after all relationships are processed
        ogm_instance.save()
        return ogm_instance

    @classmethod
    def to_pydantic(
            cls,
            ogm_instance: OGM_Model,
            pydantic_class: Optional[Type[BaseModel]] = None,
            processed_objects: Optional[Dict[int, BaseModel]] = None,
            max_depth: int = 10,
            current_path: Optional[Set[int]] = None
    ) -> Optional[BaseModel]:
        """
        Convert a neomodel OGM model instance to a Pydantic model instance.

        Args:
            ogm_instance: The OGM model instance to convert
            pydantic_class: Optional target Pydantic class (resolved from registry if not provided)
            processed_objects: Dictionary of already processed objects to handle circular references
            max_depth: Maximum recursion depth for processing relationships
            current_path: Set of object IDs in current conversion path for cycle detection

        Returns:
            Converted Pydantic model instance or None if input is None
        """
        # Handle null case
        if ogm_instance is None:
            return None

        if max_depth <= 0:
            logger.info(f"Maximum recursion depth reached for {type(ogm_instance).__name__}")
            return None

        # Initialize tracking structures if any is None
        if processed_objects is None:
            processed_objects = {}
        current_path = current_path or set()

        # Get instance ID for memory-based cycle detection
        instance_id = ogm_instance.element_id

        # Return already processed instance if we've seen it before
        if instance_id in processed_objects:
            return processed_objects[instance_id]

        # Handle cycle detection - create minimal instance with just key properties
        if instance_id in current_path:
            if pydantic_class is None:
                ogm_class = type(ogm_instance)
                if ogm_class not in cls._ogm_to_pydantic:
                    raise ConversionError(f"No mapping registered for OGM class {ogm_class.__name__}")
                pydantic_class = cls._ogm_to_pydantic[ogm_class]

            stub_instance = cls._create_minimal_pydantic_instance(ogm_instance, pydantic_class)
            processed_objects[instance_id] = stub_instance
            return stub_instance

        # Resolve Pydantic class if not provided
        if pydantic_class is None:
            ogm_class = type(ogm_instance)
            if ogm_class not in cls._ogm_to_pydantic:
                raise ConversionError(f"No mapping registered for OGM class {ogm_class.__name__}")
            pydantic_class = cls._ogm_to_pydantic[ogm_class]

        # Create namespace for type resolution
        local_namespace = {cls.__name__: cls for cls in cls._pydantic_to_ogm.keys()}
        local_namespace[pydantic_class.__name__] = pydantic_class

        # Get field types from Pydantic model
        pydantic_fields = get_type_hints(pydantic_class, globalns=None, localns=local_namespace)

        # Extract and convert properties
        pydantic_data = cls._get_property_data(ogm_instance, pydantic_fields)

        # Create a minimal instance first and register it immediately to handle circular refs
        minimal_instance = pydantic_class.model_validate(pydantic_data)
        processed_objects[instance_id] = minimal_instance

        # Add object to current path for cycle detection in nested calls
        current_path.add(instance_id)

        try:
            # Process relationships
            ogm_relationships = cls._get_ogm_relationships(type(ogm_instance))

            # Process relationships with unified approach
            for rel_name, rel in ogm_relationships.items():
                # Skip relationships not in Pydantic model
                if rel_name not in pydantic_fields:
                    continue

                # Get target class information
                target_ogm_class = cls._resolve_node_class(rel.definition.get('node_class'))
                if not target_ogm_class:
                    logger.warning(f"Could not resolve target class for relationship {rel_name}")
                    continue

                target_pydantic_class = cls._ogm_to_pydantic.get(target_ogm_class)
                if not target_pydantic_class:
                    logger.warning(f"No Pydantic model registered for OGM class {target_ogm_class.__name__}")
                    continue

                # Determine relationship cardinality
                field_type = pydantic_fields.get(rel_name)
                is_list = cls._is_list_type(field_type)
                is_single = not is_list

                # If we can't determine from field type, fall back to OGM cardinality check
                if field_type is Any:
                    cardinality_name = None
                    if hasattr(rel, 'manager'):
                        cardinality_name = rel.manager.__name__
                    elif 'cardinality' in rel.definition and hasattr(rel.definition['cardinality'], '__name__'):
                        cardinality_name = rel.definition['cardinality'].__name__

                    is_single = cardinality_name in (
                        'ZeroOrOne', 'One', 'AsyncZeroOrOne', 'AsyncOne') if cardinality_name else False

                # Get related objects
                try:
                    rel_mgr = getattr(ogm_instance, rel_name)
                    rel_objects = list(rel_mgr.all())
                except Exception as e:
                    logger.warning(f"Error retrieving relationship {rel_name}: {e}")
                    rel_objects = []

                # Convert related objects
                objects_to_process: List[StructuredNode] = [
                    rel_objects[0]] if is_single and rel_objects else rel_objects
                converted_objects = []

                for obj in objects_to_process:
                    # Skip conversion if already processed
                    obj_id = obj.element_id
                    if obj_id in processed_objects:
                        converted_objects.append(processed_objects[obj_id])
                        continue

                    # Process the related object
                    conv = cls.to_pydantic(
                        obj,
                        target_pydantic_class,
                        processed_objects,
                        max_depth - 1,
                        current_path
                    )

                    if conv is not None:
                        converted_objects.append(conv)

                # Set attribute based on cardinality
                if is_single:
                    setattr(minimal_instance, rel_name, converted_objects[0] if converted_objects else None)
                else:
                    setattr(minimal_instance, rel_name, converted_objects)

            return minimal_instance
        finally:
            # Always remove object from path when done processing
            current_path.remove(instance_id)

    @classmethod
    def _get_ogm_properties_dict(cls, ogm_instance: OGM_Model) -> dict:
        """Extract all available properties from an OGM instance into a dictionary."""
        sentinel = object()
        return {
            prop_name: value
            for prop_name, prop in cls._get_ogm_properties(type(ogm_instance)).items()
            if (value := getattr(ogm_instance, prop_name, sentinel)) is not sentinel
        }

    @classmethod
    def ogm_to_dict(
            cls,
            ogm_instance: OGM_Model,
            processed_objects: Optional[Dict[int, dict]] = None,
            max_depth: int = 10,
            current_path: Optional[List[int]] = None,
            include_properties: bool = True,
            include_relationships: bool = True
    ) -> Optional[dict]:
        """
        Convert a neomodel OGM model instance to a Python dictionary.

        This function recursively converts an OGM model (including its relationships)
        into a dictionary. It handles circular references by tracking objects already
        seen in the current conversion path.

        Args:
            ogm_instance: The OGM model instance to convert
            processed_objects: Dictionary of already processed objects to handle circular references
            max_depth: Maximum recursion depth for nested relationships
            current_path: List of object IDs in the current recursion path for cycle detection
            include_properties: Whether to include node properties in output
            include_relationships: Whether to include relationships in output

        Returns:
            A dictionary representation of the OGM instance or None if input is None
        """
        if ogm_instance is None:
            return None

        processed_objects = processed_objects or {}
        current_path = current_path or []
        instance_id = ogm_instance.element_id

        # Return already processed objects to prevent duplicate work
        if instance_id in processed_objects:
            return processed_objects[instance_id]

        # If we've reached a cycle or max depth, return only basic properties
        # This breaks circular references and prevents infinite recursion
        if any([instance_id in current_path, max_depth <= 0]):
            result = cls._get_ogm_properties_dict(ogm_instance)
            processed_objects[instance_id] = result
            return result

        # Add current object to path to detect cycles during recursion
        current_path.append(instance_id)

        # Extract properties if requested, otherwise use empty dict
        result = {} if not include_properties else cls._get_ogm_properties_dict(ogm_instance)
        processed_objects[instance_id] = result

        # Process relationships if requested and depth limit not reached
        if include_relationships and max_depth > 0:
            for rel_name, rel in cls._get_ogm_relationships(type(ogm_instance)).items():
                target_cls = cls._resolve_node_class(rel.definition.get('node_class'))
                if target_cls is None:
                    continue

                # Determine if relationship is one-to-one or one-to-many
                is_single = hasattr(rel, 'manager') and rel.manager.__name__ in (
                    'ZeroOrOne', 'One', 'AsyncZeroOrOne', 'AsyncOne'
                )

                rel_mgr = getattr(ogm_instance, rel_name, None)
                rel_objs = list(rel_mgr.all()) if rel_mgr is not None else []

                # Handle single relationships (return object or None)
                if is_single:
                    result[rel_name] = None if not rel_objs else cls.ogm_to_dict(
                        rel_objs[0], processed_objects, max_depth - 1, current_path.copy()
                    )
                # Handle collection relationships (return list of objects)
                else:
                    result[rel_name] = [
                        obj_dict for obj in rel_objs
                        if (obj_dict := cls.ogm_to_dict(
                            obj, processed_objects, max_depth - 1, current_path.copy()
                        )) is not None
                    ]

        # Remove object from current path as we're done processing it
        current_path.remove(instance_id)
        return result

    @classmethod
    def batch_dict_to_ogm(
            cls,
            data_dicts: List[dict],
            ogm_class: Type[OGM_Model],
            max_depth: int = 10
    ) -> List[OGM_Model]:
        """
        Batch convert a list of dictionaries to OGM model instances.
        """
        processed_objects: Dict[int, OGM_Model] = {}
        result: List[OGM_Model] = []

        with db.transaction:
            for d in data_dicts:
                ogm_instance = cls.dict_to_ogm(d, ogm_class, processed_objects, max_depth)
                if ogm_instance is not None:
                    result.append(ogm_instance)

        return result

    @classmethod
    def batch_ogm_to_dict(
            cls,
            ogm_instances: List[OGM_Model],
            max_depth: int = 10,
            include_properties: bool = True,
            include_relationships: bool = True
    ) -> List[dict]:
        """
        Batch convert a list of OGM model instances to dictionaries.
        """
        processed_objects: Dict[int, dict] = {}
        result: List[dict] = []

        for instance in ogm_instances:
            dict_result = cls.ogm_to_dict(
                instance,
                processed_objects,
                max_depth,
                [],
                include_properties,
                include_relationships
            )
            if dict_result is not None:
                result.append(dict_result)

        return result
