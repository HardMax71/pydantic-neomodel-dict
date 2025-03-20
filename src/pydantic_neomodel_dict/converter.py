"""
Pydantic to Neo4j OGM Converter.

This module provides a utility for converting between Pydantic models and Neo4j OGM models
with support for relationships, nested models, and custom type conversions.
"""
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    get_origin,
    get_type_hints,
)

from neomodel import (
    AsyncOne,
    AsyncZeroOrOne,
    CardinalityViolation,
    One,
    RelationshipManager,
    StructuredNode,
    ZeroOrOne,
    db,
)
from pydantic import BaseModel

from .errors import ConversionError

# Type variables for generic typing
PydanticModel = TypeVar('PydanticModel', bound=BaseModel)
OGM_Model = TypeVar('OGM_Model', bound=StructuredNode)

# HINT: Presumably, StructuredNode.element_id is str, src: ..\venv\Lib\site-packages\neo4j\graph\__init__.py

# Types for type converters
S = TypeVar("S")
T = TypeVar("T")

# Configure logger
logger = logging.getLogger(__name__)


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
            source_type: Type[S],
            target_type: Type[T],
            converter_func: Callable[[S], T]
    ) -> None:
        """
        Register a custom type converter function.

        Args:
            source_type (Type[S]): The source type to convert from.
            target_type (Type[T]): The target type to convert to.
            converter_func (Callable[[S], T]): A function that converts a value from source_type to target_type.

        Returns:
            None.
        """
        cls._type_converters[(source_type, target_type)] = converter_func
        logger.debug(f"Registered type converter: {source_type.__name__} -> {target_type.__name__}")

    @classmethod
    def _convert_value(cls, value: Optional[S], target_type: Type[T]) -> Any:
        """
        Convert the given value to the specified target type using registered converters if available.

        Args:
            value (S): The value to convert.
            target_type (Type[T]): The target type to which the value should be converted.

        Returns:
            Any: The converted value.
        """
        source_type: type[S] | type[None] = type(value)

        # Check for direct registered converter
        # Useful for objects/nested structures with objects
        converter = cls._type_converters.get((source_type, target_type))
        if converter:
            return converter(value)

        # If we get here, just return the original value
        return value

    @classmethod
    def register_models(cls, pydantic_class: Type[PydanticModel], ogm_class: Type[OGM_Model]) -> None:
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
    def stable_hash(cls, obj: Any) -> int:
        # TODO: find out better hash(), this one works but disgusting
        return hash(str(obj))

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
                if cls.stable_hash(item) not in seen_ids:
                    seen_ids.add(cls.stable_hash(item))
                    processed_list.append(item)
            value = processed_list
        pydantic_data[field_name] = value

    @classmethod
    def _get_or_create_ogm_instance(cls, data: dict, ogm_class: Type[OGM_Model]) -> OGM_Model:
        """
        Get an existing OGM instance or create a new one based on provided data.

        This method uses a hierarchical approach to find existing nodes:
        1. First try matching on unique indexed properties
        2. Then try matching on required properties
        3. Finally, use all non-None properties

        Args:
            data: Dictionary containing property values
            ogm_class: The OGM model class to use

        Returns:
            An existing (updated) or new OGM instance
        """
        defined = ogm_class.defined_properties(rels=False, aliases=False)

        # First try to match using unique indexed properties
        unique_filter = {
            k: data[k] for k in data
            if k in defined and data[k] is not None and getattr(defined[k], 'unique_index', False)
        }

        # If no unique properties available, try required properties
        required_filter = {} if unique_filter else {
            k: data[k] for k in data
            if k in defined and data[k] is not None and getattr(defined[k], 'required', False)
        }

        # If no unique or required properties, use all non-None properties
        all_props_filter = {} if (unique_filter or required_filter) else {
            k: data[k] for k in data if k in defined and data[k] is not None
        }

        # Use the most specific filter available
        filter_data = unique_filter or required_filter or all_props_filter

        # Try to find an existing node
        existing: List[OGM_Model] = ogm_class.nodes.filter(**filter_data) if filter_data else []
        if existing:
            instance: OGM_Model = existing[0]
            logger.info(f"Using existing {ogm_class.__name__} node with properties: {filter_data}")
        else:
            instance = ogm_class()
            logger.info(f"Created new {ogm_class.__name__} node with properties: {filter_data}")

            cls._set_ogm_attrs_and_save_model(data, instance)
        return instance

    @classmethod
    def to_ogm(
            cls,
            pydantic_instance: BaseModel,
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
        if max_depth <= 0:
            logger.info(f"Maximum recursion depth reached for {type(pydantic_instance).__name__}")
            return None

        processed_objects = processed_objects or {}

        instance_id = cls.stable_hash(pydantic_instance)
        if instance_id in processed_objects:
            return processed_objects[instance_id]

        if ogm_class is None:
            pydantic_class = type(pydantic_instance)
            if pydantic_class not in cls._pydantic_to_ogm:
                raise ConversionError(f"No mapping registered for Pydantic class {pydantic_class.__name__}")
            ogm_class = cls._pydantic_to_ogm[pydantic_class]

        # Create the OGM instance.
        ogm_instance: OGM_Model = ogm_class()
        processed_objects[instance_id] = ogm_instance

        pydantic_data = cls.extract_pydantic_data(pydantic_instance)
        ogm_instance = cls._get_or_create_ogm_instance(pydantic_data, ogm_class)
        processed_objects[instance_id] = ogm_instance

        # Process relationships if we have depth remaining.
        relationships = ogm_class.defined_properties(aliases=False, rels=True, properties=False)
        common_attrs = set(relationships) & set(pydantic_instance.model_fields)

        for rel_name in common_attrs:
            rel_data = getattr(pydantic_instance, rel_name)
            if not rel_data:
                continue

            target_ogm_class = relationships[rel_name].definition['node_class']
            items = cls._normalize_to_list(rel_data)

            for item in items:
                cls._process_related_item(
                    item,
                    ogm_instance,
                    rel_name,
                    target_ogm_class,
                    processed_objects,
                    max_depth - 1,
                    cls.stable_hash(pydantic_instance)
                )

        ogm_instance.save()
        return ogm_instance

    @classmethod
    def extract_pydantic_data(cls, pydantic_instance: BaseModel) -> Dict[str, Any]:
        """
        Extract data from a Pydantic model instance into a dictionary.

        This method attempts to convert a Pydantic instance to a dictionary using the model's
        built-in model_dump method. If that fails due to circular references, it falls back to
        manually processing each field individually.

        Args:
            pydantic_instance (BaseModel): The Pydantic model instance to extract data from.

        Returns:
            Dict[str, Any]: A dictionary containing all the extractable data from the Pydantic model.

        Note:
            When circular references are detected in the Pydantic model, the standard model_dump
            method raises a ValueError. In that case, this method handles each field separately
            using the _process_pydantic_field helper method, which properly manages BaseModel
            instances and lists of BaseModels to avoid circular reference issues.
        """
        # Extract Pydantic data.
        pydantic_data: Dict[str, Any] = {}
        try:
            pydantic_data = pydantic_instance.model_dump(exclude_unset=True)
        except ValueError:
            # Detected circular dependency
            # Source: https://github.com/pydantic/pydantic-core/blob/53bdfa62abefe061575d51cdb9d59b72000295ee/src/serializers/extra.rs#L183-L197
            for field_name in pydantic_instance.model_fields.keys():
                cls._process_pydantic_field(pydantic_instance, field_name, pydantic_data)
        return pydantic_data

    @classmethod
    def _process_related_item(
            cls,
            item: BaseModel,
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
            item: The item to process (BaseModel)
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
        # Handle self–references: if the item is the same as the parent, simply connect.
        if cls.stable_hash(item) == instance_id:
            rel_manager.connect(ogm_instance)
            return True

        related_instance = cls.to_ogm(
            item,
            target_ogm_class,
            processed_objects,
            max_depth
        )
        if related_instance:
            cls._connect_related_instance(rel_manager, related_instance)
            return True
        return False

    @classmethod
    def _create_minimal_pydantic_instance(
            cls,
            ogm_instance: OGM_Model,
            pydantic_class: Type[BaseModel]
    ) -> BaseModel:
        """
        Create a minimal Pydantic instance with only essential properties.
        Used for cycle breaking and max depth handling.
        """
        # Extract essential properties
        sentinel = object()
        ogm_properties = type(ogm_instance).defined_properties(rels=False, aliases=False)
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
        ogm_properties = type(ogm_instance).defined_properties(rels=False, aliases=False)

        return {
            prop_name: cls._convert_value(value, pydantic_fields.get(prop_name, Any))
            for prop_name, prop in ogm_properties.items()
            if prop_name in pydantic_fields
            if (value := getattr(ogm_instance, prop_name, sentinel)) is not sentinel
        }

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
        # Use a single processed_objects dictionary for the entire batch
        processed_objects: Dict[str, BaseModel] = {}

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

        Raises:
            ConversionError: If relationship data is not properly formatted
        """
        for rel_name, rel in ogm_relationships.items():
            if rel_name not in data_dict or data_dict[rel_name] is None:
                continue

            rel_data = data_dict[rel_name]
            # Validate relationship data type - must be dict or list
            if not isinstance(rel_data, (dict, list)):
                raise ConversionError(
                    f"Relationship '{rel_name}' must be a dictionary or list of dictionaries, "
                    f"got {type(rel_data).__name__}"
                )

            target_ogm_class = rel.definition['node_class']
            rel_manager = getattr(ogm_instance, rel_name)

            items = cls._normalize_to_list(rel_data)

            new_max_depth = max_depth - 1
            for i, item in enumerate(items):
                # First checking of relationship is valid (has to be dict), if not - raising error
                if not isinstance(item, dict):
                    raise ConversionError(
                        f"Relationship '{rel_name}' list item {i} must be a dictionary, "
                        f"got {type(item).__name__}"
                    )

                # If relationship seems to be correct - try to convert it too
                related_instance = cls.dict_to_ogm(item, target_ogm_class, processed_objects, new_max_depth)
                if related_instance:
                    cls._connect_related_instance(rel_manager, related_instance)

    @classmethod
    def _normalize_to_list(cls, rel_data: Any) -> List[Any]:
        """
        Normalize input data to a list format.

        This utility method ensures that relationship data is always in list format
        for consistent processing, regardless of whether the input is a single item
        or already a list.

        Args:
            rel_data (Any): The relationship data to normalize. Can be a single item or a list.

        Returns:
            List[Any]: A list containing the input data. If the input was already a list,
                      it is returned unchanged. If it was a single item, it is wrapped in a list.
        """
        items = rel_data if isinstance(rel_data, list) else [rel_data]
        return items

    @classmethod
    def _connect_related_instance(cls, rel_manager: RelationshipManager, related_instance: OGM_Model) -> None:
        """Helper method to connect a related instance to a relationship manager"""
        if isinstance(rel_manager, (ZeroOrOne, One, AsyncZeroOrOne, AsyncOne)):
            # For One/ZeroOrOne relationships, we need special handling
            try:
                # Try to retrieve the existing node
                current_node = rel_manager.single()
            except CardinalityViolation:
                current_node = None

            if current_node is None:
                # No node exists yet, just connect
                rel_manager.connect(related_instance)
            else:
                # Replace existing relationship with new one
                rel_manager.reconnect(current_node, related_instance)
        else:
            # For ZeroOrMore/OneOrMore relationships (like orders), always connect
            # without disconnecting previous relationships
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
        if max_depth < 0:
            logger.warning("Maximum recursion depth reached during dict_to_ogm conversion")
            return None

        processed_objects = processed_objects or {}

        # Use cls.stable_hash(data_dict) for cycle detection
        instance_id = cls.stable_hash(data_dict)
        if instance_id in processed_objects:
            return processed_objects[instance_id]

        # Get/create OGM instance that will be used
        ogm_instance = cls._get_or_create_ogm_instance(data_dict, ogm_class)
        processed_objects[instance_id] = ogm_instance

        # Process relationships
        ogm_relationships = ogm_class.defined_properties(aliases=False, rels=True, properties=False)
        cls._dict_to_ogm_process_relationships(
            ogm_instance, data_dict, ogm_relationships, processed_objects, max_depth
        )

        # Final save after all relationships are processed
        ogm_instance.save()
        return ogm_instance

    @classmethod
    def _set_ogm_attrs_and_save_model(cls, data_dict: dict, ogm_instance: OGM_Model) -> None:
        """
        Set attributes on the OGM model instance from the provided data dictionary and save the instance.

        This method iterates over the intersection of the keys in the data dictionary and the defined properties
        (excluding relationships and aliases) of the OGM instance. For each matching key, it sets the corresponding
        attribute on the OGM instance with the value from the dictionary. After updating all applicable attributes,
        the method saves the OGM instance to persist the changes.

        Args:
            data_dict (dict): A dictionary containing property names and their corresponding values.
            ogm_instance (OGM_Model): The target neomodel OGM model instance to update and save.

        Returns:
            None
        """
        # Process properties, keys for which exist in both OGM and data dict
        for prop_name in ogm_instance.defined_properties(rels=False, aliases=False).keys() & data_dict.keys():
            value = data_dict[prop_name]
            setattr(ogm_instance, prop_name, value)
        # Save properties before processing relationships
        # Not doing so will lead to error about using unsaved `neomodel` node during next step:
        # Saving relationships
        ogm_instance.save()

    @classmethod
    def to_pydantic(
            cls,
            ogm_instance: OGM_Model,
            pydantic_class: Optional[Type[BaseModel]] = None,
            processed_objects: Optional[Dict[str, BaseModel]] = None,
            max_depth: int = 10,
            current_path: Optional[Set[str]] = None
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
        # Check for maximum recursion depth
        if max_depth <= 0:
            logger.info(f"Maximum recursion depth reached for {type(ogm_instance).__name__}")
            return None

        # Initialize tracking structures
        processed_objects = processed_objects or {}
        current_path = current_path or set()

        # Handle depth limit and prepare conversion
        conversion_data = cls._prepare_pydantic_conversion(
            ogm_instance, pydantic_class, processed_objects, current_path
        )

        if not isinstance(conversion_data, tuple):
            # If not a tuple, it's an early return value (None or previously processed instance)
            return conversion_data

        # Unpack conversion data
        pydantic_class, instance_id, minimal_instance = conversion_data

        # Add object to current path for cycle detection in nested calls
        current_path.add(instance_id)

        try:
            # Process relationships
            cls._process_pydantic_relationships(
                ogm_instance, minimal_instance, pydantic_class,
                processed_objects, max_depth, current_path
            )
            return minimal_instance
        finally:
            # Always remove object from path when done processing
            current_path.remove(instance_id)

    @classmethod
    def _prepare_pydantic_conversion(
            cls,
            ogm_instance: OGM_Model,
            pydantic_class: Optional[Type[BaseModel]],
            processed_objects: Dict[str, BaseModel],
            current_path: Set[str]
    ) -> Optional[BaseModel] | Tuple[Type[BaseModel], str, BaseModel]:
        """
        Prepare for Pydantic conversion by handling common checks and initializations.

        This method handles:
        - Maximum depth check
        - Cycle detection
        - Resolving Pydantic class
        - Creating the initial Pydantic instance

        Returns:
            - A BaseModel instance for early returns (None or previously processed)
            - A tuple of (pydantic_class, instance_id, minimal_instance) for normal processing
        """
        # Get instance ID for memory-based cycle detection
        instance_id: str = ogm_instance.element_id

        # Return already processed instance if we've seen it before (not in a cycle)
        if instance_id in processed_objects and instance_id not in current_path:
            return processed_objects[instance_id]

        # Resolve Pydantic class if not provided
        if pydantic_class is None:
            ogm_class = type(ogm_instance)
            if ogm_class not in cls._ogm_to_pydantic:
                raise ConversionError(f"No mapping registered for OGM class {ogm_class.__name__}")
            pydantic_class = cls._ogm_to_pydantic[ogm_class]

        # Handle cycle detection - create minimal instance with just key properties
        if instance_id in current_path:
            # Create a new stub instance for this cycle instance
            # Important: we DO NOT store this in processed_objects to keep them distinct
            return cls._create_minimal_pydantic_instance(ogm_instance, pydantic_class)

        # Get type hints for the Pydantic model
        pydantic_type_hints = cls._resolve_pydantic_type_hints(pydantic_class)

        # Create Pydantic instance from OGM data
        minimal_instance = cls._create_pydantic_instance(
            ogm_instance, pydantic_class, pydantic_type_hints
        )

        # Register the new instance in processed objects
        processed_objects[instance_id] = minimal_instance

        # Return data needed for further processing
        return pydantic_class, instance_id, minimal_instance

    @classmethod
    def _resolve_pydantic_type_hints(cls, pydantic_class: Type[BaseModel]) -> Dict[str, Any]:
        """
        Resolve type hints for a Pydantic model class.

        Creates a local namespace for proper type resolution and returns the type hints.

        Args:
            pydantic_class: The Pydantic model class

        Returns:
            Dictionary of field names to their type annotations
        """
        # Create namespace for type resolution
        # Need this weird magic cause `neomodel` saves all Nodes in sort of global dict for whatever reason
        local_namespace = {cls.__name__: cls for cls in cls._pydantic_to_ogm.keys()}
        local_namespace[pydantic_class.__name__] = pydantic_class

        # Get field types from Pydantic model
        return get_type_hints(pydantic_class, globalns=None, localns=local_namespace)

    @classmethod
    def _create_pydantic_instance(
            cls,
            ogm_instance: OGM_Model,
            pydantic_class: Type[BaseModel],
            pydantic_type_hints: Dict[str, Any]
    ) -> BaseModel:
        """
        Create a Pydantic model instance from an OGM instance's property data.

        Args:
            ogm_instance: Source OGM model instance
            pydantic_class: Target Pydantic model class
            pydantic_type_hints: Type hints for the Pydantic model fields

        Returns:
            A new Pydantic model instance with properties set from the OGM instance
        """
        # Extract and convert properties
        pydantic_data = cls._get_property_data(ogm_instance, pydantic_type_hints)

        # Create instance without validation, compatible with both Pydantic v1 and v2
        return pydantic_class.model_construct(**pydantic_data)

    @classmethod
    def _process_pydantic_relationships(
            cls,
            ogm_instance: OGM_Model,
            pydantic_instance: BaseModel,
            pydantic_class: Type[BaseModel],
            processed_objects: Dict[str, BaseModel],
            max_depth: int,
            current_path: Set[str]
    ) -> None:
        """
        Process relationships from OGM to Pydantic model.

        Args:
            ogm_instance: Source OGM model instance
            pydantic_instance: Target Pydantic model instance to update
            pydantic_class: The Pydantic model class
            processed_objects: Dictionary of already processed objects
            max_depth: Maximum recursion depth
            current_path: Set of object IDs in the current path
        """
        # Get type hints for proper relationship handling
        pydantic_type_hints = cls._resolve_pydantic_type_hints(pydantic_class)

        # Get relationships defined in the OGM model
        ogm_relationships = type(ogm_instance).defined_properties(
            aliases=False, rels=True, properties=False
        )

        # Process each relationship
        for rel_name, rel in ogm_relationships.items():
            # Skip relationships not in Pydantic model
            if rel_name not in pydantic_type_hints:
                continue

            # Process this specific relationship
            cls._process_single_relationship(
                ogm_instance, pydantic_instance, rel_name, rel,
                pydantic_type_hints, processed_objects, max_depth, current_path
            )

    @classmethod
    def _process_single_relationship(
            cls,
            ogm_instance: OGM_Model,
            pydantic_instance: BaseModel,
            rel_name: str,
            rel: Any,
            pydantic_type_hints: Dict[str, Any],
            processed_objects: Dict[str, BaseModel],
            max_depth: int,
            current_path: Set[str]
    ) -> None:
        """
        Process a single relationship from OGM to Pydantic model.

        Args:
            ogm_instance: Source OGM model instance
            pydantic_instance: Target Pydantic model instance to update
            rel_name: Name of the relationship
            rel: Relationship definition
            pydantic_type_hints: Type hints for the Pydantic model fields
            processed_objects: Dictionary of already processed objects
            max_depth: Maximum recursion depth
            current_path: Set of object IDs in the current path
        """
        # Get target class information
        target_ogm_class = rel.definition['node_class']
        target_pydantic_class = cls._ogm_to_pydantic.get(target_ogm_class)
        if not target_pydantic_class:
            raise ConversionError(f"No Pydantic model registered for OGM class {target_ogm_class.__name__}")

        # Determine relationship cardinality
        field_type = pydantic_type_hints.get(rel_name)
        is_single = not any([get_origin(field_type) is list, field_type is list])

        # Get related objects
        rel_mgr = getattr(ogm_instance, rel_name)
        rel_objects = list(rel_mgr.all())

        # Convert related objects
        converted_objects = [
            cls.to_pydantic(
                obj,
                target_pydantic_class,
                processed_objects,
                max_depth - 1,
                current_path
            )
            for obj in rel_objects
        ]

        # Set attribute based on cardinality
        if is_single:
            setattr(pydantic_instance, rel_name, converted_objects[0] if converted_objects else None)
        else:
            setattr(pydantic_instance, rel_name, converted_objects)

    @classmethod
    def _get_ogm_properties_dict(cls, ogm_instance: OGM_Model) -> dict:
        """Extract all available properties from an OGM instance into a dictionary."""
        sentinel = object()
        return {
            prop_name: value
            for prop_name, prop in type(ogm_instance).defined_properties(rels=False, aliases=False).items()
            if (value := getattr(ogm_instance, prop_name, sentinel)) is not sentinel
        }

    @classmethod
    def ogm_to_dict(
            cls,
            ogm_instance: OGM_Model,
            processed_objects: Optional[Dict[str, dict]] = None,
            max_depth: int = 10,
            current_path: Optional[Set[str]] = None,
            include_properties: bool = True,
            include_relationships: bool = True,
    ) -> Optional[dict]:
        """
        Convert a neomodel OGM model instance to a Python dictionary.

        Args:
            ogm_instance: The OGM model instance to convert
            processed_objects: Dictionary of already processed objects to handle circular references
            max_depth: Maximum recursion depth for nested relationships
            current_path: Set of object IDs in the current recursion path for cycle detection
            include_properties: Whether to include node properties in output
            include_relationships: Whether to include relationships in output

        Returns:
            A dictionary representation of the OGM instance or None if input is None

        Conversion rules:
          - For ONE relationships (determined by the relationship manager’s type):
              if no related node exists, return None;
              if a related node exists, convert and return its dict.
          - For MANY relationships:
              if no related nodes exist, return None;
              if exactly one related node exists, convert and return its dict;
              if more than one related node exists, convert each and return as a list of dicts.
          - For this specific conversion type, if a pair of OGM and Pydantic model available in conversion dict,
              instead of returning None for all non-specified values, algorithm will try to check for hints from
              Pydantic models first and if such are available - return empty collections not as None but as []/{}.
        """
        processed_objects = processed_objects or {}
        current_path = current_path or set()
        instance_id: str = ogm_instance.element_id

        if instance_id in current_path:
            return cls._get_ogm_properties_dict(ogm_instance)
        if instance_id in processed_objects.keys():
            return processed_objects[instance_id]
        if max_depth <= 0:
            result = cls._get_ogm_properties_dict(ogm_instance)
            processed_objects[instance_id] = result
            return result

        current_path.add(instance_id)
        result = cls._get_ogm_properties_dict(ogm_instance) if include_properties else {}
        processed_objects[instance_id] = result

        if include_relationships:
            for rel_name, rel in type(ogm_instance).defined_properties(aliases=False, rels=True,
                                                                       properties=False).items():
                result[rel_name] = cls._process_relationship(current_path, include_properties,
                                                             max_depth, ogm_instance, processed_objects,
                                                             rel, rel_name)

        current_path.remove(instance_id)
        return result

    @classmethod
    def _process_relationship(
            cls,
            current_path: Set[str],
            include_properties: bool,
            max_depth: int,
            ogm_instance: OGM_Model,
            processed_objects: Dict[str, dict],
            rel: Any,
            rel_name: str
    ) -> Any:
        is_single: bool = hasattr(rel, 'manager') and rel.manager.__name__ in (
            'ZeroOrOne', 'One', 'AsyncZeroOrOne', 'AsyncOne'
        )
        rel_mgr: Optional[RelationshipManager] = getattr(ogm_instance, rel_name, None)
        rel_objs: List[OGM_Model] = cls.get_related_ogms(rel_mgr)

        # Capturing special case for 0/1-1 relationship first
        if is_single:
            return (cls.ogm_to_dict(
                rel_objs[0],
                processed_objects,
                max_depth - 1,
                current_path.copy(),
                include_properties,
                include_relationships=True
            ) if rel_objs else None)

        # Else - checking for 0/1-* relationships
        # First - if #(relationships) = 0/1
        if len(rel_objs) <= 1:
            value: Optional[dict] = None if not rel_objs else cls.ogm_to_dict(
                rel_objs[0],
                processed_objects,
                max_depth - 1,
                current_path.copy(),
                include_properties,
                include_relationships=True
            )
            pyd_cls: Optional[Type[BaseModel]] = cls._ogm_to_pydantic.get(type(ogm_instance))
            field_type: Optional[type] = get_type_hints(pyd_cls).get(rel_name) if pyd_cls else None
            return cls.process_field_value(field_type, value)

        # Then - if #(relationships) = any
        converted_list: List[dict] = []
        for obj in rel_objs:
            obj_dict = cls.ogm_to_dict(
                obj,
                processed_objects,
                max_depth - 1,
                current_path.copy(),
                include_properties,
                include_relationships=True
            )
            if obj_dict is not None:
                converted_list.append(obj_dict)
        return converted_list

    @classmethod
    def process_field_value(cls, field_type: Optional[type], value: Optional[Any]) -> Any:
        """
        Process a field value based on the expected type from the Pydantic model.

        If `value` is None, return an empty collection if the expected type is a collection (list/dict),
        otherwise return None.

        If `value` is not None and the expected type is a list (or its origin is list), wrap the value in a list.
        Otherwise, return the value as is.
        """
        origin = get_origin(field_type)
        is_list_type = origin is list or field_type is list
        is_dict_type = origin is dict or field_type is dict

        match (is_list_type, is_dict_type, value is None):
            case (True, _, True):
                return []
            case (True, _, False):
                return [value]
            case (_, True, True):
                return {}
            case _:
                return value

    @classmethod
    def get_related_ogms(cls, rel_mgr: Optional[RelationshipManager]) -> List[OGM_Model]:
        """Tries to return all related objects to given manager, if any exist. If none - returns []"""
        try:
            rel_objs = list(rel_mgr.all()) if rel_mgr is not None else []
        except CardinalityViolation:
            # Will be thrown if there's 1-1 connection, but object on either side is missing
            rel_objs = []
        return rel_objs

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
        processed_objects: Dict[str, dict] = {}
        result: List[dict] = []

        for instance in ogm_instances:
            dict_result = cls.ogm_to_dict(
                instance,
                processed_objects,
                max_depth,
                set(),
                include_properties,
                include_relationships
            )
            if dict_result is not None:
                result.append(dict_result)

        return result
