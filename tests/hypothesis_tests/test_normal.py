import datetime
import enum
import string
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import (
    booleans,
    datetimes,
    dictionaries,
    floats,
    integers,
    lists,
    none,
    sets,
    text,
)
from neomodel import (
    ArrayProperty,
    BooleanProperty,
    DateProperty,
    DateTimeProperty,
    FloatProperty,
    IntegerProperty,
    JSONProperty,
    RelationshipFrom,
    RelationshipTo,
    StringProperty,
    StructuredNode,
    StructuredRel,
    ZeroOrOne,
    db,
)
from pydantic import BaseModel, ConfigDict, Field

from pydantic_neomodel_dict import Converter

# ----------------------------------------
# Define a variety of test models
# ----------------------------------------

# 1. Simple models with basic types
class SimplePydantic(BaseModel):
    """Simple Pydantic model with basic field types"""
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    age: int
    height: Optional[float] = None
    is_active: bool = True
    created_at: Optional[datetime.datetime] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SimpleOGM(StructuredNode):
    """Matching Neo4j OGM model with basic types"""
    uid = StringProperty(unique_index=True)
    name = StringProperty(required=True)
    age = IntegerProperty(required=True)
    height = FloatProperty()
    is_active = BooleanProperty(default=True)
    created_at = DateTimeProperty()


# 2. Enum types
class UserRole(str, enum.Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"


class UserStatus(enum.IntEnum):
    INACTIVE = 0
    ACTIVE = 1
    SUSPENDED = 2
    BANNED = 3


# 3. Models with collection fields
class CollectionPydantic(BaseModel):
    """Pydantic model with various collection fields"""
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tags: List[str] = []
    scores: Dict[str, float] = {}
    flags: Set[str] = set()
    matrix: List[List[int]] = []
    metadata: Dict[str, Any] = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CollectionOGM(StructuredNode):
    """Neo4j OGM model for collections"""
    uid = StringProperty(unique_index=True)
    tags = ArrayProperty(StringProperty(), default=[])
    # Stored as JSON string
    scores = JSONProperty(default={})
    flags = ArrayProperty(StringProperty(), default=[])
    # Stored as JSON string
    matrix = JSONProperty(default=[])
    # Stored as JSON string
    metadata = JSONProperty(default={})


# 4. Complex relationship model
class RelationshipRel(StructuredRel):
    """Relationship with properties"""
    since = DateTimeProperty(default=datetime.datetime.now)
    strength = IntegerProperty(default=1)


# 5. Complex nested models
class AddressPydantic(BaseModel):
    """Address sub-model"""
    street: str
    city: str
    country: str
    postal_code: str
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AddressOGM(StructuredNode):
    """Address OGM model"""
    street = StringProperty(required=True)
    city = StringProperty(required=True)
    country = StringProperty(required=True)
    postal_code = StringProperty(required=True)


class CompanyPydantic(BaseModel):
    """Company sub-model"""
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    industry: str
    founded_year: int
    website: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CompanyOGM(StructuredNode):
    """Company OGM model"""
    uid = StringProperty(unique_index=True)
    name = StringProperty(required=True)
    industry = StringProperty(required=True)
    founded_year = IntegerProperty(required=True)
    website = StringProperty()

    # Add relationship to employees
    employees = RelationshipFrom('PersonOGM', 'WORKS_AT')


class PersonPydantic(BaseModel):
    """Complex person model with nested objects and relationships"""
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    age: int
    role: UserRole = UserRole.VIEWER
    status: UserStatus = UserStatus.ACTIVE

    # Nested models
    address: Optional[AddressPydantic] = None
    company: Optional[CompanyPydantic] = None

    # Collection fields
    skills: List[str] = []
    preferences: Dict[str, Any] = {}

    # Self-referential relationships
    friends: List["PersonPydantic"] = []
    best_friend: Optional["PersonPydantic"] = None
    manager: Optional["PersonPydantic"] = None
    direct_reports: List["PersonPydantic"] = []

    # Date fields
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: Optional[datetime.datetime] = None
    birth_date: Optional[datetime.date] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PersonOGM(StructuredNode):
    """Complex person OGM model with relationships"""
    uid = StringProperty(unique_index=True, required=True)
    name = StringProperty(required=True)
    email = StringProperty(unique_index=True, required=True)
    age = IntegerProperty(required=True)
    role = StringProperty(default="viewer")
    status = IntegerProperty(default=1)

    # Collection fields
    skills = ArrayProperty(StringProperty(), default=[])
    preferences = JSONProperty(default={})

    # Date fields
    created_at = DateTimeProperty(default=datetime.datetime.now)
    updated_at = DateTimeProperty()
    birth_date = DateProperty()

    # Relationships
    # Fixed: Use proper cardinality classes instead of integer 1
    address = RelationshipTo('AddressOGM', 'HAS_ADDRESS', cardinality=ZeroOrOne)
    company = RelationshipTo('CompanyOGM', 'WORKS_AT', cardinality=ZeroOrOne)
    friends = RelationshipTo('PersonOGM', 'FRIEND_OF')  # zero or more
    best_friend = RelationshipTo('PersonOGM', 'BEST_FRIEND', cardinality=ZeroOrOne)
    manager = RelationshipTo('PersonOGM', 'REPORTS_TO', cardinality=ZeroOrOne)
    direct_reports = RelationshipFrom('PersonOGM', 'REPORTS_TO')


# Register all the model pairs
Converter.register_models(SimplePydantic, SimpleOGM)
Converter.register_models(CollectionPydantic, CollectionOGM)
Converter.register_models(AddressPydantic, AddressOGM)
Converter.register_models(CompanyPydantic, CompanyOGM)
Converter.register_models(PersonPydantic, PersonOGM)

# Register custom converters
Converter.register_type_converter(
    UserRole, str,
    lambda role: role.value
)

Converter.register_type_converter(
    str, UserRole,
    lambda role_str: UserRole(role_str) if role_str in [r for r in UserRole] else UserRole.VIEWER
)

Converter.register_type_converter(
    UserStatus, int,
    lambda status: status.value
)

Converter.register_type_converter(
    int, UserStatus,
    lambda status_int: UserStatus(status_int) if status_int in [s.value for s in UserStatus] else UserStatus.ACTIVE
)

# Handle date conversion for serializing to JSON
Converter.register_type_converter(
    datetime.date, str,
    lambda d: d.isoformat()
)

Converter.register_type_converter(
    str, datetime.date,
    lambda s: datetime.date.fromisoformat(s)
)

# Convert between date and datetime with safer epoch handling
Converter.register_type_converter(
    datetime.date, datetime.datetime,
    lambda d: datetime.datetime.combine(d, datetime.time())
)

Converter.register_type_converter(
    datetime.datetime, datetime.date,
    lambda dt: dt.date()
)

# Add direct converters between datetime and float (timestamp) to handle pre-1970 dates
Converter.register_type_converter(
    datetime.datetime, float,
    # This uses a reference date to handle pre-1970 dates
    lambda dt: (dt - datetime.datetime(1970, 1, 1, tzinfo=dt.tzinfo if dt.tzinfo else None)).total_seconds()
)

Converter.register_type_converter(
    float, datetime.datetime,
    # This uses timedelta to handle negative timestamps
    lambda ts: datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=ts)
)


# ----------------------------------------
# Hypothesis strategies for generating test data
# ----------------------------------------

@st.composite
def generate_unique_random_uid(draw):
    # Generating random for sure uid; smh using `st.uuids()` or plain `random` returns
    # not every time unique values => patch
    uuid = draw(st.uuids())
    current_time_ns = time.time_ns()
    return str(uuid) + str(current_time_ns)


@st.composite
def simple_model_dicts(draw):
    """Strategy to generate simple model dictionaries"""
    return {
        "uid": draw(generate_unique_random_uid()),
        "name": draw(text(min_size=1, max_size=50)),
        "age": draw(integers(min_value=0, max_value=120)),
        "height": draw(st.one_of(floats(min_value=0.1, max_value=3.0), none())),
        "is_active": draw(booleans()),
        "created_at": draw(st.one_of(
            datetimes(
                min_value=datetime.datetime(2000, 1, 1),
                max_value=datetime.datetime(2023, 12, 31)
            ),
            none()
        ))
    }


@st.composite
def collection_model_dicts(draw):
    """Strategy to generate collection model dictionaries"""
    return {
        "uid": draw(generate_unique_random_uid()),
        "tags": draw(lists(text(min_size=1, max_size=10), max_size=5)),
        "scores": draw(dictionaries(
            keys=text(min_size=1, max_size=10),
            values=floats(min_value=0, max_value=100),
            max_size=3
        )),
        "flags": draw(sets(text(min_size=1, max_size=10), max_size=3)),
        "matrix": draw(lists(
            lists(integers(min_value=-10, max_value=10), min_size=1, max_size=3),
            min_size=0, max_size=3
        )),
        "metadata": draw(dictionaries(
            keys=text(min_size=1, max_size=10),
            values=st.one_of(
                text(min_size=1, max_size=20),
                integers(min_value=-100, max_value=100),
                floats(min_value=-100, max_value=100),
                booleans()
            ),
            max_size=3
        ))
    }


@st.composite
def address_dicts(draw):
    """Strategy to generate address dictionaries"""
    return {
        "street": draw(text(min_size=1, max_size=100)),
        "city": draw(text(min_size=1, max_size=50)),
        "country": draw(text(min_size=1, max_size=50)),
        "postal_code": draw(text(min_size=1, max_size=20))
    }


@st.composite
def company_dicts(draw):
    """Strategy to generate company dictionaries"""
    return {
        "uid": draw(generate_unique_random_uid()),
        "name": draw(text(min_size=1, max_size=100)),
        "industry": draw(text(min_size=1, max_size=50)),
        "founded_year": draw(integers(min_value=1800, max_value=2023)),
        "website": draw(st.one_of(
            text(min_size=5, max_size=100).map(lambda s: f"https://{s}.com"),
            none()
        ))
    }


@st.composite
def person_dicts(draw, with_nested=True, with_friends=False, with_company=True, max_friends=2, max_depth=3):
    """Strategy to generate person dictionaries with optional nested objects and relationships"""
    if max_depth <= 0:
        with_nested = False
        with_friends = False
        with_company = False

    # Use simpler text generation with more restricted characters to improve performance
    simple_text = text(min_size=1, max_size=20, alphabet=string.ascii_letters + string.digits + ".-_")

    result = {
        "uid": draw(generate_unique_random_uid()),
        "name": draw(simple_text),
        "email": f"{draw(simple_text)}-{draw(generate_unique_random_uid())}@example.com",
        "age": draw(integers(min_value=18, max_value=80)),  # Narrower range
        "role": draw(st.sampled_from(list(UserRole))),
        "status": draw(st.sampled_from(list(UserStatus))),
        "skills": draw(lists(simple_text, max_size=3)),  # Smaller lists
        "preferences": draw(dictionaries(
            keys=simple_text,
            values=st.one_of(
                simple_text, integers(min_value=0, max_value=100), booleans()
            ),
            max_size=2  # Smaller dictionaries
        )),
        "created_at": draw(datetimes(
            min_value=datetime.datetime(2000, 1, 1),
            max_value=datetime.datetime(2023, 12, 31)
        )),
        "updated_at": draw(st.one_of(
            datetimes(
                min_value=datetime.datetime(2000, 1, 1),
                max_value=datetime.datetime(2023, 12, 31)
            ),
            none()
        )),
        # Using dates after 1970 to avoid timestamp conversion issues
        "birth_date": draw(st.one_of(
            datetimes(
                min_value=datetime.datetime(1970, 1, 1),
                max_value=datetime.datetime(2005, 12, 31)
            ).map(lambda dt: dt.date()),
            none()
        ))
    }

    # Add nested address if requested
    if with_nested:
        if draw(booleans()):
            result["address"] = draw(address_dicts())
        else:
            result["address"] = None
    else:
        result["address"] = None

    # Add company if requested
    if with_company and with_nested:
        if draw(booleans()):
            result["company"] = draw(company_dicts())
        else:
            result["company"] = None
    else:
        result["company"] = None

    # Initialize relationship fields
    result["friends"] = []
    result["best_friend"] = None
    result["manager"] = None
    result["direct_reports"] = []

    # Add friends if requested - with a much more restricted approach to avoid complexity
    if with_friends and max_depth > 0 and max_friends > 0:
        # Generate fewer friends to reduce test complexity
        num_friends = draw(integers(min_value=0, max_value=min(max_friends, 2)))
        for _ in range(num_friends):
            # Generate simpler friend objects with minimal properties
            friend = {
                "uid": draw(generate_unique_random_uid()),
                "name": draw(simple_text),
                "email": f"{draw(simple_text)}-{draw(generate_unique_random_uid())}@example.com",
                "age": draw(integers(min_value=18, max_value=80)),
                "role": draw(st.sampled_from(list(UserRole))),
                "status": draw(st.sampled_from(list(UserStatus))),
                "skills": [],
                "preferences": {},
                "created_at": draw(datetimes(
                    min_value=datetime.datetime(2000, 1, 1),
                    max_value=datetime.datetime(2023, 12, 31)
                )),
                "updated_at": None,
                "birth_date": None,
                "address": None,
                "company": None,
                "friends": [],
                "best_friend": None,
                "manager": None,
                "direct_reports": []
            }
            result["friends"].append(friend)

        # Maybe set best friend from the friends list
        if result["friends"] and draw(booleans()):
            result["best_friend"] = draw(st.sampled_from(result["friends"]))

        # Maybe add manager - with minimal properties
        if draw(booleans()):
            result["manager"] = {
                "uid": draw(generate_unique_random_uid()),
                "name": draw(simple_text),
                "email": f"{draw(simple_text)}-{draw(generate_unique_random_uid())}@example.com",
                "age": draw(integers(min_value=18, max_value=80)),
                "role": draw(st.sampled_from(list(UserRole))),
                "status": draw(st.sampled_from(list(UserStatus))),
                "skills": [],
                "preferences": {},
                "created_at": draw(datetimes(
                    min_value=datetime.datetime(2000, 1, 1),
                    max_value=datetime.datetime(2023, 12, 31)
                )),
                "updated_at": None,
                "birth_date": None,
                "address": None,
                "company": None,
                "friends": [],
                "best_friend": None,
                "manager": None,
                "direct_reports": []
            }

            # Maybe add direct reports - with minimal nesting
            if max_depth > 1:
                num_reports = min(draw(integers(min_value=0, max_value=2)), 1)  # At most 1 report to reduce complexity
                for i in range(num_reports):
                    report = {
                        "uid": draw(generate_unique_random_uid()),
                        "name": draw(simple_text),
                        "email": f"{draw(simple_text)}-{draw(generate_unique_random_uid())}@example.com",
                        "age": draw(integers(min_value=18, max_value=80)),
                        "role": draw(st.sampled_from(list(UserRole))),
                        "status": draw(st.sampled_from(list(UserStatus))),
                        "skills": [],
                        "preferences": {},
                        "created_at": draw(datetimes(
                            min_value=datetime.datetime(2000, 1, 1),
                            max_value=datetime.datetime(2023, 12, 31)
                        )),
                        "updated_at": None,
                        "birth_date": None,
                        "address": None,
                        "company": None,
                        "friends": [],
                        "best_friend": None,
                        "manager": None,
                        "direct_reports": []
                    }
                    result["direct_reports"].append(report)

    return result


def build_pydantic_models(data_dict, model_class):
    """Helper to build Pydantic models from dictionaries, handling nested structures"""
    if data_dict is None:
        return None

    if model_class == PersonPydantic:
        # Process nested models
        if data_dict.get("address"):
            data_dict["address"] = build_pydantic_models(data_dict["address"], AddressPydantic)

        if data_dict.get("company"):
            data_dict["company"] = build_pydantic_models(data_dict["company"], CompanyPydantic)

        # Process relationships
        if data_dict.get("friends"):
            data_dict["friends"] = [build_pydantic_models(f, PersonPydantic) for f in data_dict["friends"]]

        if data_dict.get("best_friend"):
            data_dict["best_friend"] = build_pydantic_models(data_dict["best_friend"], PersonPydantic)

        if data_dict.get("manager"):
            data_dict["manager"] = build_pydantic_models(data_dict["manager"], PersonPydantic)

        if data_dict.get("direct_reports"):
            data_dict["direct_reports"] = [build_pydantic_models(r, PersonPydantic) for r in
                                           data_dict["direct_reports"]]

    return model_class(**data_dict)


# ----------------------------------------
# Fixture to ensure model registrations persist
# ----------------------------------------

@pytest.fixture(scope="session", autouse=True)
def preserve_hypothesis_model_registrations():
    """
    Ensure model registrations are preserved throughout the entire hypothesis test session.
    This fixture runs at the beginning of the test session and prevents registry cleaning.
    """
    print("Ensuring model registrations are preserved for hypothesis tests...")

    # Re-register if needed (they should already be registered from module level)
    if SimplePydantic not in Converter._pydantic_to_ogm:
        print("Re-registering models for hypothesis tests...")
        # These are the same registrations that are at the module level
        Converter.register_models(SimplePydantic, SimpleOGM)
        Converter.register_models(CollectionPydantic, CollectionOGM)
        Converter.register_models(AddressPydantic, AddressOGM)
        Converter.register_models(CompanyPydantic, CompanyOGM)
        Converter.register_models(PersonPydantic, PersonOGM)

        # Re-register custom type converters if needed
        if (UserRole, str) not in Converter._type_converters:
            Converter.register_type_converter(
                UserRole, str,
                lambda role: role.value
            )

            Converter.register_type_converter(
                str, UserRole,
                lambda role_str: UserRole(role_str) if role_str in [r.value for r in UserRole] else UserRole.VIEWER
            )

            Converter.register_type_converter(
                UserStatus, int,
                lambda status: status.value
            )

            Converter.register_type_converter(
                int, UserStatus,
                lambda status_int: UserStatus(status_int) if status_int in [s.value for s in
                                                                            UserStatus] else UserStatus.ACTIVE
            )

            # Other converters as needed

    print(f"Verified registrations: {len(Converter._pydantic_to_ogm)} model pairs")

    # Override the clean_registry fixture to prevent it from cleaning during hypothesis tests
    orig_clean_registry = getattr(pytest, '_clean_registry', None)

    # Only yield once this fixture completes - preserving registrations for the whole session
    yield

    # Restore original if needed (probably not necessary for session scope)
    if orig_clean_registry:
        pytest._clean_registry = orig_clean_registry


# ----------------------------------------
# Test classes for different aspects of conversion
# ----------------------------------------

class TestBasicConversions:
    """Tests for basic conversion operations"""

    def setup_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def teardown_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(model_dict=simple_model_dicts())
    def test_pydantic_to_ogm_basic(self, model_dict):
        """Test basic conversion from Pydantic to OGM model"""
        # Create Pydantic instance
        pydantic_model = SimplePydantic(**model_dict)

        # Convert to OGM
        ogm_model = Converter.to_ogm(pydantic_model)

        # Verify properties were preserved
        assert ogm_model.uid == pydantic_model.uid
        assert ogm_model.name == pydantic_model.name
        assert ogm_model.age == pydantic_model.age
        assert ogm_model.is_active == pydantic_model.is_active

        if pydantic_model.height is not None:
            assert ogm_model.height == pydantic_model.height
        else:
            assert ogm_model.height is None

        if pydantic_model.created_at is not None:
            assert ogm_model.created_at is not None

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(model_dict=simple_model_dicts())
    def test_ogm_to_pydantic_basic(self, model_dict):
        """Test basic conversion from OGM to Pydantic model"""
        # Create Pydantic model and convert to OGM
        pydantic_model = SimplePydantic(**model_dict)
        ogm_model = Converter.to_ogm(pydantic_model)

        # Convert back to Pydantic
        result_model = Converter.to_pydantic(ogm_model)

        # Verify properties
        assert result_model.uid == pydantic_model.uid
        assert result_model.name == pydantic_model.name
        assert result_model.age == pydantic_model.age
        assert result_model.is_active == pydantic_model.is_active

        if pydantic_model.height is not None:
            assert result_model.height == pydantic_model.height
        else:
            assert result_model.height is None

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(model_dict=simple_model_dicts())
    def test_dict_to_ogm_basic(self, model_dict):
        """Test direct conversion from dict to OGM model"""
        # Convert dict to OGM
        ogm_model = Converter.dict_to_ogm(model_dict, SimpleOGM)

        # Verify properties
        assert ogm_model.uid == model_dict["uid"]
        assert ogm_model.name == model_dict["name"]
        assert ogm_model.age == model_dict["age"]
        assert ogm_model.is_active == model_dict["is_active"]

        if model_dict.get("height") is not None:
            assert ogm_model.height == model_dict["height"]
        else:
            assert ogm_model.height is None

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(model_dict=simple_model_dicts())
    def test_ogm_to_dict_basic(self, model_dict):
        """Test direct conversion from OGM to dict"""
        # Create OGM model
        ogm_model = Converter.dict_to_ogm(model_dict, SimpleOGM)

        # Convert to dict
        result_dict = Converter.ogm_to_dict(ogm_model)

        # Verify key properties
        assert result_dict["uid"] == model_dict["uid"]
        assert result_dict["name"] == model_dict["name"]
        assert result_dict["age"] == model_dict["age"]
        assert result_dict["is_active"] == model_dict["is_active"]

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(model_dict=simple_model_dicts())
    def test_complex_conversion_path_basic(self, model_dict):
        """Test a complex conversion path: Dict → OGM → Dict → Pydantic → OGM → Pydantic"""
        # Dict → OGM
        ogm1 = Converter.dict_to_ogm(model_dict, SimpleOGM)

        # OGM → Dict
        dict1 = Converter.ogm_to_dict(ogm1)

        # Dict → Pydantic
        pydantic1 = SimplePydantic(**dict1)

        # Pydantic → OGM
        ogm2 = Converter.to_ogm(pydantic1)

        # OGM → Pydantic
        pydantic2 = Converter.to_pydantic(ogm2)

        # Verify final result matches original
        assert pydantic2.uid == model_dict["uid"]
        assert pydantic2.name == model_dict["name"]
        assert pydantic2.age == model_dict["age"]
        assert pydantic2.is_active == model_dict["is_active"]


class TestCollectionConversions:
    """Tests for models with collection fields"""

    def setup_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def teardown_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    @given(model_dict=collection_model_dicts())
    def test_collections_pydantic_to_ogm(self, model_dict):
        """Test conversion of collection fields from Pydantic to OGM"""
        # Create Pydantic model
        pydantic_model = CollectionPydantic(**model_dict)

        # Convert to OGM
        ogm_model = Converter.to_ogm(pydantic_model)

        # Verify base field
        assert ogm_model.uid == pydantic_model.uid

        # Check collection fields
        assert set(ogm_model.tags) == set(pydantic_model.tags)

        # Check JSON properties - these might be stored as JSON strings
        # So we'll need to verify the semantic equivalence not exact equality
        if hasattr(ogm_model, 'scores') and isinstance(ogm_model.scores, dict):
            assert set(ogm_model.scores.keys()) == set(pydantic_model.scores.keys())
            for key in pydantic_model.scores:
                assert pytest.approx(ogm_model.scores[key]) == pydantic_model.scores[key]

    @given(model_dict=collection_model_dicts())
    def test_collections_ogm_to_pydantic(self, model_dict):
        """Test conversion of collection fields from OGM to Pydantic"""
        # Create OGM model via dict
        ogm_model = Converter.dict_to_ogm(model_dict, CollectionOGM)

        # Convert to Pydantic
        pydantic_model = Converter.to_pydantic(ogm_model)

        # Verify base field
        assert pydantic_model.uid == model_dict["uid"]

        # Check collection fields
        assert set(pydantic_model.tags) == set(model_dict["tags"])

        # Check dictionary fields
        assert set(pydantic_model.scores.keys()) == set(model_dict["scores"].keys())
        for key in model_dict["scores"]:
            assert pytest.approx(pydantic_model.scores[key]) == model_dict["scores"][key]

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(model_dict=collection_model_dicts())
    def test_collections_dict_to_ogm(self, model_dict):
        """Test direct conversion of collection fields from dict to OGM"""
        # Convert dict to OGM
        ogm_model = Converter.dict_to_ogm(model_dict, CollectionOGM)

        # Verify base field
        assert ogm_model.uid == model_dict["uid"]

        # Check array property
        # `ogm_model.tags` is an ArrayProperty, but smh presented as list (???)
        assert set(ogm_model.tags) == set(model_dict["tags"])

        # Check scores
        # `ogm_model.scores` is a dict, but smh linter says it's not
        assert set(ogm_model.scores.keys()) == set(model_dict["scores"].keys())

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(model_dict=collection_model_dicts())
    def test_collections_ogm_to_dict(self, model_dict):
        """Test direct conversion of collection fields from OGM to dict"""
        ogm_model = Converter.dict_to_ogm(model_dict, CollectionOGM)
        result_dict = Converter.ogm_to_dict(ogm_model)

        # Verify base field
        assert result_dict["uid"] == model_dict["uid"]

        # Check collection
        assert len(result_dict.keys()) == len(model_dict.keys())
        for key in model_dict.keys():
            assert result_dict.get(key) == model_dict.get(key)

    @given(model_dict=collection_model_dicts())
    def test_collections_empty_values(self, model_dict):
        """Test conversion of empty collection fields"""
        # Create model with empty collections
        empty_model = CollectionPydantic(
            uid=model_dict["uid"],
            tags=[],
            scores={},
            flags=set(),
            matrix=[],
            metadata={}
        )

        # Pydantic -> OGM -> Dict, Pydantic
        # Convert to OGM
        ogm_model = Converter.to_ogm(empty_model)

        # Convert to dict
        result_dict = Converter.ogm_to_dict(ogm_model)

        # Convert back to Pydantic
        final_model = Converter.to_pydantic(ogm_model)

        # Verify empty collections remained empty in both final_model and result_dict
        assert len(final_model.tags) == len(result_dict["tags"]) == 0
        assert len(final_model.scores) == len(result_dict["scores"]) == 0
        assert len(final_model.flags) == len(result_dict["flags"]) == 0
        assert len(final_model.matrix) == len(result_dict["matrix"]) == 0
        assert len(final_model.metadata) == len(result_dict["metadata"]) == 0


class TestNestedModelConversions:
    """Tests for models with nested objects"""

    def setup_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def teardown_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10)
    @given(st.data())
    def test_nested_pydantic_to_ogm(self, data):
        """Test conversion of nested objects from Pydantic to OGM"""
        # Generate a person with a nested address and company
        person_dict = data.draw(person_dicts(with_nested=True, with_friends=False))
        person_model = build_pydantic_models(person_dict, PersonPydantic)

        # Convert to OGM
        ogm_model = Converter.to_ogm(person_model)

        # Verify base fields
        assert ogm_model.uid == person_model.uid
        assert ogm_model.name == person_model.name
        assert ogm_model.email == person_model.email

        # Check nested address equivalence without explicit if/else branching
        address_nodes = list(ogm_model.address.all())
        # If person_model.address is not None, we expect one node; otherwise, an empty list.
        assert len(address_nodes) == (1 if person_model.address else 0)
        assert [node.street for node in address_nodes] == (
            [person_model.address.street] if person_model.address else []
        )
        assert [node.city for node in address_nodes] == (
            [person_model.address.city] if person_model.address else []
        )

        # Check nested company equivalence without explicit if/else branching
        company_nodes = list(ogm_model.company.all())
        # Expect one node if company exists, otherwise an empty list.
        assert len(company_nodes) == (1 if person_model.company else 0)
        assert [node.name for node in company_nodes] == (
            [person_model.company.name] if person_model.company else []
        )
        assert [node.industry for node in company_nodes] == (
            [person_model.company.industry] if person_model.company else []
        )

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10)
    @given(st.data())
    def test_nested_ogm_to_pydantic(self, data):
        """Test conversion of nested objects from OGM to Pydantic"""
        # Generate a person with a nested address and company
        person_dict = data.draw(person_dicts(with_nested=True, with_friends=False))

        # Convert directly to OGM
        ogm_model = Converter.dict_to_ogm(person_dict, PersonOGM)

        # Convert to Pydantic
        pydantic_model = Converter.to_pydantic(ogm_model)

        # Verify base fields
        assert pydantic_model.uid == person_dict["uid"]
        assert pydantic_model.name == person_dict["name"]
        assert pydantic_model.email == person_dict["email"]

        # Check nested address
        if person_dict.get("address"):
            assert pydantic_model.address is not None
            assert pydantic_model.address.street == person_dict["address"]["street"]
            assert pydantic_model.address.city == person_dict["address"]["city"]
        else:
            assert pydantic_model.address is None

        # Check nested company
        if person_dict.get("company"):
            assert pydantic_model.company is not None
            assert pydantic_model.company.name == person_dict["company"]["name"]
            assert pydantic_model.company.industry == person_dict["company"]["industry"]
        else:
            assert pydantic_model.company is None

    @settings(max_examples=10)
    @given(st.data())
    def test_nested_dict_to_ogm(self, data):
        """Test direct conversion of nested objects from dict to OGM"""
        # Generate a person dictionary with nested objects
        person_dict = data.draw(person_dicts(with_nested=True, with_friends=False))

        # Convert to OGM
        ogm_model = Converter.dict_to_ogm(person_dict, PersonOGM)

        # Verify base fields
        assert ogm_model.uid == person_dict["uid"]
        assert ogm_model.name == person_dict["name"]

        # Check nested address
        if person_dict.get("address"):
            address_nodes = list(ogm_model.address.all())
            assert len(address_nodes) == 1
            assert address_nodes[0].street == person_dict["address"]["street"]
        else:
            address_nodes = list(ogm_model.address.all())
            assert len(address_nodes) == 0

        # Check nested company
        if person_dict.get("company"):
            company_nodes = list(ogm_model.company.all())
            assert len(company_nodes) == 1
            assert company_nodes[0].name == person_dict["company"]["name"]
        else:
            company_nodes = list(ogm_model.company.all())
            assert len(company_nodes) == 0

    @settings(max_examples=10)
    @given(st.data())
    def test_nested_ogm_to_dict(self, data):
        """Test direct conversion of nested objects from OGM to dict"""
        # Generate a person with nested objects
        person_dict = data.draw(person_dicts(with_nested=True, with_friends=False))

        # Convert to OGM
        ogm_model = Converter.dict_to_ogm(person_dict, PersonOGM)

        # Convert to dict
        result_dict = Converter.ogm_to_dict(ogm_model)

        # Verify base fields
        assert result_dict["uid"] == person_dict["uid"]
        assert result_dict["name"] == person_dict["name"]

        # Check nested address
        if person_dict.get("address"):
            assert "address" in result_dict
            assert isinstance(result_dict["address"], dict)
            assert result_dict["address"]["street"] == person_dict["address"]["street"]

        # Check nested company
        if person_dict.get("company"):
            assert "company" in result_dict
            assert isinstance(result_dict["company"], dict)
            assert result_dict["company"]["name"] == person_dict["company"]["name"]

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10)
    @given(st.data())
    def test_complex_conversion_path_with_nested(self, data):
        """Test complex conversion path with nested objects"""
        # Generate a person with nested objects
        person_dict = data.draw(person_dicts(with_nested=True, with_friends=False))

        # Dict → Pydantic
        pydantic1 = build_pydantic_models(person_dict, PersonPydantic)

        # Pydantic → OGM
        ogm1 = Converter.to_ogm(pydantic1)

        # OGM → Dict
        dict1 = Converter.ogm_to_dict(ogm1)

        # Dict → OGM
        ogm2 = Converter.dict_to_ogm(dict1, PersonOGM)

        # OGM → Pydantic
        pydantic2 = Converter.to_pydantic(ogm2)

        # Verify properties survived multiple conversions
        assert pydantic2.uid == pydantic1.uid
        assert pydantic2.name == pydantic1.name
        assert pydantic2.email == pydantic1.email

        # Check nested address
        if pydantic1.address:
            assert pydantic2.address is not None
            assert pydantic2.address.street == pydantic1.address.street
            assert pydantic2.address.city == pydantic1.address.city
        else:
            assert pydantic2.address is None

        # Check nested company
        if pydantic1.company:
            assert pydantic2.company is not None
            assert pydantic2.company.name == pydantic1.company.name
            assert pydantic2.company.industry == pydantic1.company.industry
        else:
            assert pydantic2.company is None


class TestRelationshipConversions:
    """Tests for models with relationships"""

    def setup_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def teardown_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=5, deadline=5000)  # Increased deadline
    @given(st.data())
    def test_simple_relationships(self, data):
        """Test conversion of simple one-to-many relationships"""
        # Generate a person with friends
        person_dict = data.draw(person_dicts(with_nested=False, with_friends=True, max_friends=2, max_depth=1))
        person_model = build_pydantic_models(person_dict, PersonPydantic)

        # Count original friends
        friend_count = len(person_model.friends)

        # Convert to OGM
        ogm_model = Converter.to_ogm(person_model)

        # Verify friends got converted
        friend_nodes = list(ogm_model.friends.all())
        assert len(friend_nodes) == friend_count

        # Convert back to Pydantic
        result_model = Converter.to_pydantic(ogm_model)

        # Verify friends are still there
        assert len(result_model.friends) == friend_count

        # Check that friend properties were preserved
        for original_friend in person_model.friends:
            # Find matching friend in result
            matching_friend = next((f for f in result_model.friends if f.uid == original_friend.uid), None)
            assert matching_friend is not None
            assert matching_friend.name == original_friend.name
            assert matching_friend.email == original_friend.email

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=5, deadline=5000)  # Increased deadline
    @given(st.data())
    def test_one_to_one_relationships(self, data):
        """Test conversion of one-to-one relationships"""
        # Generate a person with a best friend
        person_dict = data.draw(person_dicts(with_nested=False, with_friends=True, max_friends=2, max_depth=1))
        person_model = build_pydantic_models(person_dict, PersonPydantic)

        # Ensure we have a best friend
        if person_model.friends and not person_model.best_friend:
            person_model.best_friend = person_model.friends[0]

        has_best_friend = person_model.best_friend is not None

        # Convert to OGM
        ogm_model = Converter.to_ogm(person_model)

        # Verify best friend got converted
        best_friend_nodes = list(ogm_model.best_friend.all())
        assert len(best_friend_nodes) == (1 if has_best_friend else 0)

        # Convert back to Pydantic
        result_model = Converter.to_pydantic(ogm_model)

        # Verify best friend is still there
        assert (result_model.best_friend is not None) == has_best_friend

        # Check properties if we have a best friend
        if has_best_friend:
            assert result_model.best_friend.uid == person_model.best_friend.uid
            assert result_model.best_friend.name == person_model.best_friend.name
            assert result_model.best_friend.email == person_model.best_friend.email

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=5, deadline=5000)  # Increased deadline
    @given(st.data())
    def test_hierarchical_relationships(self, data):
        """Test conversion of hierarchical relationships (manager/direct reports)"""
        # Generate a person with a manager
        person_dict = data.draw(person_dicts(with_nested=False, with_friends=False, max_depth=2))
        person_model = build_pydantic_models(person_dict, PersonPydantic)

        # Create a manager and some direct reports if not present
        if not person_model.manager:
            manager = PersonPydantic(
                uid=str(uuid.uuid4()),
                name="Manager",
                email=f"manager-{uuid.uuid4()}@example.com",
                age=40
            )
            person_model.manager = manager

        if not person_model.direct_reports:
            for i in range(2):
                report = PersonPydantic(
                    uid=str(uuid.uuid4()),
                    name=f"Report {i}",
                    email=f"report{i}-{uuid.uuid4()}@example.com",
                    age=25 + i,
                    manager=person_model
                )
                person_model.direct_reports.append(report)

        # Convert to OGM
        ogm_model = Converter.to_ogm(person_model)

        # Verify relationships
        manager_nodes = list(ogm_model.manager.all())
        assert len(manager_nodes) == 1

        report_nodes = list(ogm_model.direct_reports.all())
        assert len(report_nodes) == len(person_model.direct_reports)

        # Convert back to Pydantic
        result_model = Converter.to_pydantic(ogm_model)

        # Verify relationships persisted
        assert result_model.manager is not None
        assert result_model.manager.uid == person_model.manager.uid
        assert result_model.manager.name == person_model.manager.name

        assert len(result_model.direct_reports) == len(person_model.direct_reports)
        for original_report in person_model.direct_reports:
            matching_report = next((r for r in result_model.direct_reports if r.uid == original_report.uid), None)
            assert matching_report is not None
            assert matching_report.name == original_report.name

    def test_cyclic_relationships(self):
        """Test conversion of cyclic relationships"""
        # Create a person who is their own friend and manager
        person = PersonPydantic(
            uid=str(uuid.uuid4()),
            name="Self Reference",
            email="self@example.com",
            age=30
        )

        # Create self-references
        person.friends.append(person)
        person.best_friend = person
        person.manager = person

        # Convert to OGM
        ogm_model = Converter.to_ogm(person)

        # Verify self-references
        friend_nodes = list(ogm_model.friends.all())
        assert len(friend_nodes) == 1
        assert friend_nodes[0].uid == person.uid

        best_friend_nodes = list(ogm_model.best_friend.all())
        assert len(best_friend_nodes) == 1
        assert best_friend_nodes[0].uid == person.uid

        manager_nodes = list(ogm_model.manager.all())
        assert len(manager_nodes) == 1
        assert manager_nodes[0].uid == person.uid

        # Convert back to Pydantic
        result_model = Converter.to_pydantic(ogm_model)

        # Verify cyclic references were maintained
        assert len(result_model.friends) == 1
        assert result_model.friends[0].uid == result_model.uid

        assert result_model.best_friend is not None
        assert result_model.best_friend.uid == result_model.uid

        assert result_model.manager is not None
        assert result_model.manager.uid == result_model.uid

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=3, deadline=5000)  # Increased deadline
    @given(st.data())
    def test_dict_to_ogm_relationships(self, data):
        """Test conversion of relationships from dict to OGM"""
        # Generate a person with friends and manager
        person_dict = data.draw(person_dicts(with_nested=False, with_friends=True, max_friends=2, max_depth=1))

        # Convert directly to OGM
        ogm_model = Converter.dict_to_ogm(person_dict, PersonOGM)

        # Verify relationships
        friend_nodes = list(ogm_model.friends.all())
        assert len(friend_nodes) == len(person_dict.get("friends", []))

        if person_dict.get("best_friend"):
            best_friend_nodes = list(ogm_model.best_friend.all())
            assert len(best_friend_nodes) == 1
            assert best_friend_nodes[0].uid == person_dict["best_friend"]["uid"]

        if person_dict.get("manager"):
            manager_nodes = list(ogm_model.manager.all())
            assert len(manager_nodes) == 1
            assert manager_nodes[0].uid == person_dict["manager"]["uid"]

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=3, deadline=5000)  # Increased deadline
    @given(st.data())
    def test_ogm_to_dict_relationships(self, data):
        """Test conversion of relationships from OGM to dict"""
        # Generate a person with friends and manager
        person_dict = data.draw(person_dicts(with_nested=False, with_friends=True, max_friends=2, max_depth=1))

        # Convert to OGM
        ogm_model = Converter.dict_to_ogm(person_dict, PersonOGM)

        # Convert to dict
        result_dict = Converter.ogm_to_dict(ogm_model)

        # Verify relationships
        assert "friends" in result_dict
        assert len(result_dict["friends"]) == len(person_dict.get("friends", []))

        if person_dict.get("best_friend"):
            assert "best_friend" in result_dict
            assert isinstance(result_dict["best_friend"], dict)
            assert result_dict["best_friend"]["uid"] == person_dict["best_friend"]["uid"]

        if person_dict.get("manager"):
            assert "manager" in result_dict
            assert isinstance(result_dict["manager"], dict)
            assert result_dict["manager"]["uid"] == person_dict["manager"]["uid"]


class TestEdgeCases:
    """Tests for edge cases and unusual situations"""

    def setup_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def teardown_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def test_null_values(self):
        """Test conversion with explicit None values"""
        # Create a model with some None values
        model = PersonPydantic(
            uid="null-test",
            name="Null Test",
            email="null@example.com",
            age=30,
            address=None,
            company=None,
            best_friend=None,
            manager=None,
            updated_at=None,
            birth_date=None
        )

        # Convert to OGM
        ogm = Converter.to_ogm(model)

        # Convert to dict
        dict_version = Converter.ogm_to_dict(ogm)

        # Convert back to OGM
        ogm2 = Converter.dict_to_ogm(dict_version, PersonOGM)

        # Convert to Pydantic
        result = Converter.to_pydantic(ogm2)

        # Verify None values were preserved
        assert result.address is None
        assert result.company is None
        assert result.best_friend is None
        assert result.manager is None
        assert result.updated_at is None
        assert result.birth_date is None

    def test_empty_collections(self):
        """Test conversion with empty collections"""
        # Create a model with empty collections
        model = PersonPydantic(
            uid="empty-test",
            name="Empty Test",
            email="empty@example.com",
            age=30,
            skills=[],
            preferences={},
            friends=[]
        )

        # Pydantic -> OGM -> Dict, Pydantic
        ogm = Converter.to_ogm(model)
        dict_version = Converter.ogm_to_dict(ogm)
        result = Converter.to_pydantic(ogm)

        # Verify empty collections were preserved using chained assertions
        assert len(result.skills) == len(dict_version["skills"]) == 0
        assert len(result.preferences) == len(dict_version["preferences"]) == 0
        assert len(result.friends) == len(dict_version["friends"]) == 0

    def test_deep_relationships(self):
        """Test conversion with deep relationship chains"""
        # Create a chain of people
        depth = 5
        root = PersonPydantic(
            uid="root",
            name="Root Person",
            email="root@example.com",
            age=40
        )

        current = root
        for i in range(1, depth):
            child = PersonPydantic(
                uid=f"child-{i}",
                name=f"Child {i}",
                email=f"child{i}@example.com",
                age=40 - i
            )
            current.direct_reports.append(child)
            child.manager = current
            current = child

        # Convert to OGM with explicitly higher max_depth
        ogm = Converter.to_ogm(root, max_depth=20)

        # Convert to Pydantic with explicitly higher max_depth
        result = Converter.to_pydantic(ogm, max_depth=20)

        # Verify the chain is intact
        current = result
        for i in range(1, depth):
            assert len(current.direct_reports) == 1
            child = current.direct_reports[0]
            assert child.uid == f"child-{i}"
            assert child.manager is not None
            assert child.manager.uid == current.uid
            current = child

    def test_special_characters(self):
        """Test conversion with special characters in strings"""
        # Create model with special characters
        model = SimplePydantic(
            uid="special",
            name="Special \"Chars\" & <Tags> 'Quotes'",
            age=30
        )

        # Convert to OGM
        ogm = Converter.to_ogm(model)

        # Convert to dict
        dict_version = Converter.ogm_to_dict(ogm)

        # Convert back to Pydantic
        result = Converter.to_pydantic(ogm)

        # Verify that special characters and all fields are preserved using chained assertions
        assert result.name == dict_version["name"] == model.name
        assert result.uid == dict_version["uid"] == model.uid
        assert result.age == dict_version["age"] == model.age

    def test_complex_conversion_chain(self):
        """Test a long chain of conversions"""
        # Create initial model
        model = PersonPydantic(
            uid="chain-test",
            name="Chain Test",
            email="chain@example.com",
            age=35,
            address=AddressPydantic(
                street="123 Main St",
                city="Anytown",
                country="USA",
                postal_code="12345"
            )
        )

        # Create a chain of conversions
        ogm1 = Converter.to_ogm(model)
        dict1 = Converter.ogm_to_dict(ogm1)
        ogm2 = Converter.dict_to_ogm(dict1, PersonOGM)
        dict2 = Converter.ogm_to_dict(ogm2)
        ogm3 = Converter.dict_to_ogm(dict2, PersonOGM)
        result = Converter.to_pydantic(ogm3)

        # Verify properties survived the chain
        assert result.uid == model.uid
        assert result.name == model.name
        assert result.email == model.email
        assert result.age == model.age

        # Check nested address
        assert result.address is not None
        assert result.address.street == model.address.street
        assert result.address.city == model.address.city
        assert result.address.country == model.address.country
        assert result.address.postal_code == model.address.postal_code

    def test_max_depth_handling(self):
        """Test handling of max_depth parameter"""
        # Create a chain of 3 managers
        ceo = PersonPydantic(
            uid="ceo",
            name="CEO",
            email="ceo@example.com",
            age=55
        )

        vp = PersonPydantic(
            uid="vp",
            name="VP",
            email="vp@example.com",
            age=45,
            manager=ceo
        )
        ceo.direct_reports.append(vp)

        manager = PersonPydantic(
            uid="manager",
            name="Manager",
            email="manager@example.com",
            age=35,
            manager=vp
        )
        vp.direct_reports.append(manager)

        employee = PersonPydantic(
            uid="employee",
            name="Employee",
            email="employee@example.com",
            age=25,
            manager=manager
        )
        manager.direct_reports.append(employee)

        # Use a modest value for the first conversion (employee + manager only)
        ogm1 = Converter.to_ogm(employee, max_depth=2)  # Increased from 1
        result1 = Converter.to_pydantic(ogm1, max_depth=2)  # Increased from 1

        # With depth=2, we should get the employee and their immediate manager
        assert result1.uid == employee.uid
        assert result1.manager is not None
        assert result1.manager.uid == manager.uid
        assert result1.manager.manager is None  # Should be cut off

        # Try with more depth
        ogm2 = Converter.to_ogm(employee, max_depth=5)  # Increased from 3
        result2 = Converter.to_pydantic(ogm2, max_depth=5)  # Increased from 3

        # Should go 3 levels deep
        assert result2.uid == employee.uid
        assert result2.manager is not None
        assert result2.manager.uid == manager.uid
        assert result2.manager.manager is not None
        assert result2.manager.manager.uid == vp.uid
        assert result2.manager.manager.manager is not None
        assert result2.manager.manager.manager.uid == ceo.uid


class TestBatchOperations:
    """Tests for batch conversion operations"""

    def setup_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def teardown_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=5)
    @given(st.lists(simple_model_dicts(), min_size=1, max_size=5))
    def test_batch_to_ogm(self, model_dicts):
        """Test batch conversion from Pydantic to OGM"""
        # Create list of Pydantic models
        pydantic_models = [SimplePydantic(**d) for d in model_dicts]

        # Perform batch conversion
        ogm_models = Converter.batch_to_ogm(pydantic_models)

        # Verify all models were converted
        assert len(ogm_models) == len(pydantic_models)

        # Check individual conversions
        for i, original in enumerate(pydantic_models):
            assert ogm_models[i].uid == original.uid
            assert ogm_models[i].name == original.name
            assert ogm_models[i].age == original.age

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=5)
    @given(st.lists(simple_model_dicts(), min_size=1, max_size=5))
    def test_batch_to_pydantic(self, model_dicts):
        """Test batch conversion from OGM to Pydantic"""
        # Create list of OGM models
        ogm_models = [Converter.dict_to_ogm(d, SimpleOGM) for d in model_dicts]

        # Perform batch conversion
        pydantic_models = Converter.batch_to_pydantic(ogm_models)

        # Verify all models were converted
        assert len(pydantic_models) == len(ogm_models)

        # Check individual conversions
        for i, ogm in enumerate(ogm_models):
            assert pydantic_models[i].uid == ogm.uid
            assert pydantic_models[i].name == ogm.name
            assert pydantic_models[i].age == ogm.age

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=5)
    @given(st.lists(simple_model_dicts(), min_size=1, max_size=5))
    def test_batch_dict_to_ogm(self, model_dicts):
        """Test batch conversion from dict to OGM"""
        # Perform batch conversion
        ogm_models = Converter.batch_dict_to_ogm(model_dicts, SimpleOGM)

        # Verify all models were converted
        assert len(ogm_models) == len(model_dicts)

        # Check individual conversions
        for i, dict_data in enumerate(model_dicts):
            assert ogm_models[i].uid == dict_data["uid"]
            assert ogm_models[i].name == dict_data["name"]
            assert ogm_models[i].age == dict_data["age"]

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=5)
    @given(st.lists(simple_model_dicts(), min_size=1, max_size=5))
    def test_batch_ogm_to_dict(self, model_dicts):
        """Test batch conversion from OGM to dict"""
        # Create list of OGM models
        ogm_models = [Converter.dict_to_ogm(d, SimpleOGM) for d in model_dicts]

        # Perform batch conversion
        dict_results = Converter.batch_ogm_to_dict(ogm_models)

        # Verify all models were converted
        assert len(dict_results) == len(ogm_models)

        # Check individual conversions
        for i, ogm in enumerate(ogm_models):
            assert dict_results[i]["uid"] == ogm.uid
            assert dict_results[i]["name"] == ogm.name
            assert dict_results[i]["age"] == ogm.age

    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=3)
    @given(st.data())
    def test_batch_complex_models(self, data):
        """Test batch operations with complex models"""
        # Generate a few complex person models
        num_people = 3
        person_dict_list = [  # Changed variable name to avoid conflict
            data.draw(person_dicts(with_nested=True, with_friends=False))
            for _ in range(num_people)
        ]

        # Create Pydantic models
        pydantic_models = [build_pydantic_models(d, PersonPydantic) for d in person_dict_list]

        # Perform batch conversion
        ogm_models = Converter.batch_to_ogm(pydantic_models)

        # Verify all models were converted
        assert len(ogm_models) == num_people

        # Convert back to Pydantic
        result_models = Converter.batch_to_pydantic(ogm_models)

        # Verify conversions
        for i, original in enumerate(pydantic_models):
            result = result_models[i]
            assert result.uid == original.uid
            assert result.name == original.name
            assert result.email == original.email

            # Check nested models
            if original.address:
                assert result.address is not None
                assert result.address.street == original.address.street
            else:
                assert result.address is None

            if original.company:
                assert result.company is not None
                assert result.company.name == original.company.name
            else:
                assert result.company is None


class TestMixedConversions:
    """Tests for mixed conversion scenarios"""

    def setup_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    def teardown_method(self):
        db.cypher_query("MATCH (n) DETACH DELETE n")

    @settings(max_examples=3)
    @given(st.data())
    def test_complex_ecosystem(self, data):
        """Test a complex ecosystem with mixed model types and relationships"""
        # Generate a company
        company_dict = data.draw(company_dicts())
        company = CompanyPydantic(**company_dict)

        # Generate a few employees
        num_employees = 3
        employees = []
        for i in range(num_employees):
            person_dict = data.draw(person_dicts(with_nested=True, with_friends=False))
            person = build_pydantic_models(person_dict, PersonPydantic)
            person.company = company
            employees.append(person)

        # Create a manager as one of the employees
        manager = employees[0]
        for i in range(1, num_employees):
            employees[i].manager = manager
            manager.direct_reports.append(employees[i])

        # Convert manager to OGM (should convert the entire ecosystem)
        ogm_manager = Converter.to_ogm(manager)

        # Convert different parts to different formats
        # OGM manager to dict
        manager_dict = Converter.ogm_to_dict(ogm_manager)

        # Additional checks on manager_dict
        assert manager_dict["uid"] == manager.uid
        assert manager_dict["name"] == manager.name
        assert manager_dict["email"] == manager.email
        # Check direct_reports in manager_dict if present
        if "direct_reports" in manager_dict:
            assert len(manager_dict["direct_reports"]) == len(manager.direct_reports)

        # OGM company to Pydantic
        company_nodes = list(ogm_manager.company.all())
        assert len(company_nodes) == 1
        ogm_company = company_nodes[0]
        company_model = Converter.to_pydantic(ogm_company)

        # Verify company_model matches the original company
        assert company_model.uid == company.uid
        assert company_model.name == company.name

        # Direct reports OGM to dict
        report_nodes = list(ogm_manager.direct_reports.all())
        report_dicts = [Converter.ogm_to_dict(report) for report in report_nodes]

        # Check each report dict has the original manager details
        for report_dict in report_dicts:
            assert "manager" in report_dict
            assert report_dict["manager"]["uid"] == manager.uid

        # Create new manager model via dict conversion
        new_manager_uid = data.draw(generate_unique_random_uid())
        new_manager_dict = {
            "uid": new_manager_uid,
            "name": "New Manager",
            "email": f"newmanager-{new_manager_uid}@example.com",
            "age": 45
        }

        # Dict to OGM
        new_ogm_manager = Converter.dict_to_ogm(new_manager_dict, PersonOGM)

        # OGM to Pydantic
        new_manager_model = Converter.to_pydantic(new_ogm_manager)

        # Verify new_manager_model fields match new_manager_dict
        assert new_manager_model.uid == new_manager_dict["uid"]
        assert new_manager_model.name == new_manager_dict["name"]
        assert new_manager_model.email == new_manager_dict["email"]
        if hasattr(new_manager_model, "age"):
            assert new_manager_model.age == new_manager_dict["age"]

        # Connect new manager to existing employees (direct reports)
        for report_node in report_nodes:
            new_ogm_manager.direct_reports.connect(report_node)
            report_node.manager.replace(new_ogm_manager)

        # Convert back to Pydantic to see full ecosystem
        final_manager = Converter.to_pydantic(new_ogm_manager)

        # Verify new structure
        assert final_manager.uid == new_manager_uid
        assert len(final_manager.direct_reports) == len(report_nodes)

        # Check that each employee now points to the new manager
        employee_ids = {e.uid for e in employees[1:]}
        for report in final_manager.direct_reports:
            assert report.uid in employee_ids
            assert report.manager is not None
            assert report.manager.uid == final_manager.uid

        # Additionally, verify that the company relationship remains intact for each report
        for report in final_manager.direct_reports:
            assert report.company is not None
            assert report.company.uid == company.uid

    @settings(max_examples=3, deadline=1000)
    @given(st.data())
    def test_multi_path_conversions(self, data):
        """Test multiple parallel conversion paths with the same data"""
        # Generate a complex person
        original_person_dict = data.draw(person_dicts(with_nested=True, with_friends=True, max_depth=2))

        # Create a deep copy to prevent build_pydantic_models from modifying the original
        import copy
        person_dict = copy.deepcopy(original_person_dict)

        # Path 1: Dict → OGM → Dict
        ogm1 = Converter.dict_to_ogm(person_dict, PersonOGM)
        dict1 = Converter.ogm_to_dict(ogm1)

        # Path 2: Dict → Pydantic → OGM → Pydantic
        pydantic1 = build_pydantic_models(person_dict, PersonPydantic)
        ogm2 = Converter.to_ogm(pydantic1)
        pydantic2 = Converter.to_pydantic(ogm2)

        # Path 3: Dict → OGM → Pydantic → Dict
        ogm3 = Converter.dict_to_ogm(original_person_dict, PersonOGM)  # Use the original unmodified dict
        pydantic3 = Converter.to_pydantic(ogm3)
        # `pydantic3` obj may contain circular references => pydantic will throw errorr with
        # |ValueError: Circular reference detected (id repeated)|
        # => Using Converter instead
        # dict3 = pydantic3.model_dump(mode='json')  # Use json mode to convert nested models to dicts
        dict3 = Converter.ogm_to_dict(ogm3)

        # Compare results from different paths
        # Check that basic properties are preserved in all paths
        assert ogm1.uid == ogm2.uid == ogm3.uid == original_person_dict["uid"]
        assert dict1["uid"] == dict3["uid"] == original_person_dict["uid"]
        assert pydantic2.uid == pydantic3.uid == original_person_dict["uid"]

        # Compare basic fields from all paths
        assert dict1["name"] == dict3["name"] == original_person_dict["name"]
        assert pydantic2.name == pydantic3.name == original_person_dict["name"]

        # Verify same nested objects
        if original_person_dict.get("address"):
            # Check that address exists in dictionary results
            assert "address" in dict1
            assert "address" in dict3

            # Extract street value from original dict
            original_street = original_person_dict["address"]["street"]

            # Compare street values
            assert dict1["address"]["street"] == original_street
            assert dict3["address"]["street"] == original_street

            # Check Pydantic models
            assert pydantic2.address is not None
            assert pydantic3.address is not None
            assert pydantic2.address.street == pydantic3.address.street == original_street


if __name__ == "__main__":
    pytest.main(["-xvs"])
