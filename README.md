<h1 align="center">Pydantic ↔ Neomodel (Neo4j) OGM ↔ Python Dict Converter</h1>

<p align="center">
  <a href="https://github.com/HardMax71/pydantic-neomodel-dict/actions/workflows/ruff.yml">
    <img src="https://github.com/HardMax71/pydantic-neomodel-dict/actions/workflows/ruff.yml/badge.svg?branch=main" alt="Ruff">
  </a>
  &nbsp;
  <a href="https://github.com/HardMax71/pydantic-neomodel-dict/actions/workflows/mypy.yml">
    <img src="https://github.com/HardMax71/pydantic-neomodel-dict/actions/workflows/mypy.yml/badge.svg?branch=main" alt="MyPy">
  </a>
  &nbsp;
  <a href="https://sonarcloud.io/dashboard?id=HardMax71_pydantic-neomodel-dict">
    <img src="https://sonarcloud.io/api/project_badges/measure?project=HardMax71_pydantic-neomodel-dict&metric=alert_status" alt="Quality Gate Status">
  </a>
</p>
<p align="center">
  <a href="https://github.com/HardMax71/pydantic-neomodel-dict/actions/workflows/tests.yml">
    <img src="https://github.com/HardMax71/pydantic-neomodel-dict/actions/workflows/tests.yml/badge.svg?branch=main" alt="Tests">
  </a>
&nbsp;
  <a href="https://codecov.io/gh/HardMax71/pydantic-neomodel-dict">
    <img src="https://codecov.io/gh/HardMax71/pydantic-neomodel-dict/branch/main/graph/badge.svg" alt="Codecov">
  </a>
</p>
<p align="center">
  <a href="https://badge.fury.io/py/pydantic-neomodel-dict">
    <img src="https://badge.fury.io/py/pydantic-neomodel-dict.svg" alt="PyPI version">
  </a>
  &nbsp;
  <a href="https://pypi.org/project/pydantic-neomodel-dict/">
    <img src="https://img.shields.io/pypi/pyversions/pydantic-neomodel-dict.svg" alt="Python versions">
  </a>
</p>

A bidirectional converter between Pydantic models, Neomodel (Neo4j) OGM (Object Graph Mapper) models, and Python dictionaries. This
library simplifies the integration between Pydantic's data validation capabilities and Neo4j's graph database
operations.

## Features

- **Bidirectional Conversion**: Convert seamlessly between Pydantic models, Neomodel (Neo4j) OGM models, and Python dictionaries
- **Relationship Handling**: Process complex relationships at any level of nesting
- **Circular Reference Support**: Detect and properly handle circular references in object graphs
- **Custom Type Conversion**: Register custom type converters for specialized data transformations
- **Batch Operations**: Convert multiple objects efficiently with transaction support
- **Type Safety**: Full typing support with `mypy`

## Installation

```bash
pip install pydantic-neomodel-dict
```

## Requirements

- Python 3.10+
- pydantic 2.0.0+
- neomodel 5.0.0+

## Basic Usage

Here's a simple example demonstrating conversion between Pydantic and Neomodel (Neo4j) OGM models:

> [!NOTE]  
> Before execution of example down here, use
> supplied [docker-compose.yml](https://github.com/HardMax71/pydantic-neomodel-dict/blob/main/docker-compose.yml)
> and start `neo4j` container inside via `docker-compose up --build`.

```python
from pydantic import BaseModel
from neomodel import StructuredNode, StringProperty, IntegerProperty, config
from pydantic_neomodel_dict import Converter

# Set up Neomodel (Neo4j) connection - this is required!
config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'
config.ENCRYPTED_CONNECTION = False
config.AUTO_INSTALL_LABELS = True


# Define your models
class UserPydantic(BaseModel):
  name: str
  email: str
  age: int


class UserOGM(StructuredNode):
  name = StringProperty(required=True)
  email = StringProperty(unique_index=True, required=True)
  age = IntegerProperty(index=True, default=0)


# Register the model mapping
Converter.register_models(UserPydantic, UserOGM)

# Convert Pydantic to OGM
user_pydantic = UserPydantic(name="John Doe", email="john@example.com", age=30)
user_ogm = Converter.to_ogm(user_pydantic)

# Convert OGM to Pydantic
user_pydantic_again = Converter.to_pydantic(user_ogm)

# Convert to/from dictionary
user_dict = {"name": "Jane Doe", "email": "jane@example.com", "age": 25}
user_ogm_from_dict = Converter.dict_to_ogm(user_dict, UserOGM)
user_dict_from_ogm = Converter.ogm_to_dict(user_ogm)

# Print results to verify
print(f"Original user: {user_pydantic}")
print(f"After round-trip conversion: {user_pydantic_again}")
print(f"Dictionary conversion result: {user_dict_from_ogm}")
```

``` 
$ python3 example.py
Original user: name='John Doe' email='john@example.com' age=30
After round-trip conversion: name='John Doe' email='john@example.com' age=30
Dictionary conversion result: {'name': 'John Doe', 'email': 'john@example.com', 'age': 30}
```

## Examples

<details>
<summary>Simple Model Conversion</summary>

This example demonstrates basic conversion between Pydantic models and Neomodel (Neo4j) OGM models:

```python
from pydantic import BaseModel
from neomodel import StructuredNode, StringProperty, IntegerProperty, UniqueIdProperty, config
from pydantic_neomodel_dict import Converter

# Set up Neomodel (Neo4j) connection - this is required!
config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'
config.ENCRYPTED_CONNECTION = False
config.AUTO_INSTALL_LABELS = True


# Define Pydantic model
class ProductPydantic(BaseModel):
  uid: str
  name: str
  price: float
  sku: str


# Define Neomodel (Neo4j) OGM model
class ProductOGM(StructuredNode):
  uid = UniqueIdProperty()
  name = StringProperty(required=True)
  price = IntegerProperty(required=True)
  sku = StringProperty(unique_index=True, required=True)


# Register the models
Converter.register_models(ProductPydantic, ProductOGM)

# Create a Pydantic instance
product = ProductPydantic(
  uid="123e4567-e89b-12d3-a456-426614174000",
  name="Wireless Headphones",
  price=99.99,
  sku="WH-X1000"
)

# Convert to Neomodel (Neo4j) OGM model
product_ogm = Converter.to_ogm(product)

# Save to database
# product_ogm is already saved during conversion

# Query from database
retrieved_product = ProductOGM.nodes.get(sku="WH-X1000")

# Convert back to Pydantic model
product_pydantic = Converter.to_pydantic(retrieved_product)

print(f"Product: {product_pydantic.name}, Price: {product_pydantic.price}")
```

Output:

``` 
Product: Wireless Headphones, Price: 99
```

</details>

<details>
<summary>Nested Relationships</summary>

This example shows how to handle nested relationships between models:

```python
import random
from typing import List

from neomodel import IntegerProperty, One, RelationshipFrom, RelationshipTo, StringProperty, StructuredNode, config
from pydantic import BaseModel

from pydantic_neomodel_dict import Converter

# Set up Neomodel (Neo4j) connection - this is required!
config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'
config.ENCRYPTED_CONNECTION = False
config.AUTO_INSTALL_LABELS = True


# Define Pydantic models
class AddressPydantic(BaseModel):
  street: str
  city: str
  zip_code: str


class OrderPydantic(BaseModel):
  order_id: str
  amount: float


class CustomerPydantic(BaseModel):
  name: str
  email: str
  address: AddressPydantic
  orders: List[OrderPydantic] = []


# Define Neomodel (Neo4j) OGM models
class AddressOGM(StructuredNode):
  street = StringProperty(required=True)
  city = StringProperty(required=True)
  zip_code = StringProperty(required=True)


class OrderOGM(StructuredNode):
  order_id = StringProperty(unique_index=True, required=True)
  amount = IntegerProperty(required=True)
  customer = RelationshipFrom('CustomerOGM', 'PLACED')


class CustomerOGM(StructuredNode):
  name = StringProperty(required=True)
  email = StringProperty(unique_index=True, required=True)
  address = RelationshipTo(AddressOGM, 'HAS_ADDRESS', One)
  orders = RelationshipTo(OrderOGM, 'PLACED')


# Register model mappings
Converter.register_models(AddressPydantic, AddressOGM)
Converter.register_models(OrderPydantic, OrderOGM)
Converter.register_models(CustomerPydantic, CustomerOGM)

# Create a customer with address and orders
email = f"jane{random.randint(1, 1000)}@example.com"
customer = CustomerPydantic(
  name="Jane Smith",
  email=email,
  address=AddressPydantic(
    street="123 Main St",
    city="New York",
    zip_code="10001"
  ),
  orders=[
    OrderPydantic(order_id="ORD-001", amount=125.50),
    OrderPydantic(order_id="ORD-002", amount=75.25)
  ]
)

# Convert to Neomodel (Neo4j) OGM model (this will create all related nodes)
customer_ogm = Converter.to_ogm(customer)

# Retrieve and convert back
retrieved_customer = CustomerOGM.nodes.get(email=email)
customer_pydantic = Converter.to_pydantic(retrieved_customer)

print(f"Customer: {customer_pydantic.name}")
print(f"Address: {customer_pydantic.address.street}, {customer_pydantic.address.city}")
print(f"Orders: {len(customer_pydantic.orders)}")
print("Whole dict: \n", customer_pydantic.model_dump())
```

Output:

```
Customer: Jane Smith
Address: 123 Main St, New York
Orders: 2
Whole dict: 
 {'name': 'Jane Smith', 'email': 'jane672@example.com', 'orders': [{'order_id': 'ORD-002', 'amount': 75}, {'order_id': 'ORD-001', 'amount': 125}], 'address': {'street': '123 Main St', 'city': 'New York', 'zip_code': '10001'}}

```

</details>

<details>
<summary>Handling Circular References</summary>

This example demonstrates how the converter handles circular references in object graphs:

```python
from typing import List

from neomodel import (
  StructuredNode, StringProperty, RelationshipTo, config
)
from pydantic import BaseModel

from pydantic_neomodel_dict import Converter

# Set up Neomodel (Neo4j) connection - this is required!
config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'
config.ENCRYPTED_CONNECTION = False
config.AUTO_INSTALL_LABELS = True


# Define Pydantic models with circular references
class PersonPydantic(BaseModel):
  name: str
  friends: List['PersonPydantic'] = []


# Add self-reference resolution
PersonPydantic.model_rebuild()


# Define Neomodel (Neo4j) OGM models
class PersonOGM(StructuredNode):
  name = StringProperty(required=True, unique_index=True)
  friends = RelationshipTo('PersonOGM', 'FRIENDS_WITH')


# Register models
Converter.register_models(PersonPydantic, PersonOGM)

# Create instances with circular references
alice = PersonPydantic(name="Alice")
bob = PersonPydantic(name="Bob")
charlie = PersonPydantic(name="Charlie")

# Create circular references
alice.friends = [bob, charlie]
bob.friends = [alice, charlie]
charlie.friends = [alice, bob]

# Convert to Neomodel (Neo4j) OGM models (handles circular references)
alice_ogm = Converter.to_ogm(alice)

# Convert back to Pydantic
alice_pydantic = Converter.to_pydantic(alice_ogm)

print(f"{alice_pydantic.name}'s friends: {[friend.name for friend in alice_pydantic.friends]}")
print(f"{alice_pydantic.friends[0].name}'s friends: {[friend.name for friend in alice_pydantic.friends[0].friends]}")
```

Output:

``` 
Alice's friends: ['Charlie', 'Bob']
Charlie's friends: ['Bob', 'Alice']
```

</details>

<details>
<summary>Custom Type Converters</summary>

This example shows how to use custom type converters for specialized data transformations:

```python
from datetime import datetime, date

from neomodel import (
  StructuredNode, StringProperty, DateProperty
)
from neomodel import (
  config
)
from pydantic import BaseModel

from pydantic_neomodel_dict import Converter

# Set up Neomodel (Neo4j) connection - this is required!
config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'
config.ENCRYPTED_CONNECTION = False
config.AUTO_INSTALL_LABELS = True


# Define models
class EventPydantic(BaseModel):
  title: str
  event_date: datetime  # Using Python datetime


class EventOGM(StructuredNode):
  title = StringProperty(required=True)
  event_date = DateProperty(required=True)  # Neomodel (Neo4j) uses date


# Register custom type converters
Converter.register_type_converter(
  datetime, date,  # Convert from datetime to date
  lambda dt: dt.date()  # Conversion function
)

Converter.register_type_converter(
  date, datetime,  # Convert from date to datetime
  lambda d: datetime.combine(d, datetime.min.time())  # Conversion function
)

# Register models
Converter.register_models(EventPydantic, EventOGM)

# Create a Pydantic instance with datetime
event = EventPydantic(
  title="Conference",
  event_date=datetime(2023, 10, 15, 9, 0, 0)
)

# Convert to Neomodel (Neo4j) OGM (datetime will be converted to date)
event_ogm = Converter.to_ogm(event)

# Convert back to Pydantic (date will be converted to datetime)
event_pydantic = Converter.to_pydantic(event_ogm)

print(f"Event: {event_pydantic.title}")
print(f"Date: {event_pydantic.event_date}")
print(f"Type: {type(event_pydantic.event_date)}")
print("Whole object:\n", event_pydantic.model_dump())
```

Output:

``` 
Event: Conference
Date: 2023-10-15 09:00:00
Type: <class 'datetime.datetime'>
Whole object:
 {'title': 'Conference', 'event_date': datetime.datetime(2023, 10, 15, 9, 0)}
```

</details>

<details>
<summary>Batch Operations</summary>

This example demonstrates batch conversion of multiple objects:

```python
from neomodel import StructuredNode, StringProperty, IntegerProperty, config
from pydantic import BaseModel

from pydantic_neomodel_dict import Converter

# Set up Neomodel (Neo4j) connection - this is required!
config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'
config.ENCRYPTED_CONNECTION = False
config.AUTO_INSTALL_LABELS = True


# Define models
class ProductPydantic(BaseModel):
  name: str
  sku: str
  price: float
  inventory: int


class ProductOGM(StructuredNode):
  name = StringProperty(required=True)
  sku = StringProperty(unique_index=True, required=True)
  price = IntegerProperty(required=True)
  inventory = IntegerProperty(default=0)


# Register models
Converter.register_models(ProductPydantic, ProductOGM)

# Create multiple Pydantic instances
products = [
  ProductPydantic(name="Laptop", sku="LT-001", price=1299.99, inventory=10),
  ProductPydantic(name="Smartphone", sku="SP-002", price=899.99, inventory=15),
  ProductPydantic(name="Headphones", sku="HP-003", price=199.99, inventory=25),
  ProductPydantic(name="Tablet", sku="TB-004", price=499.99, inventory=8),
  ProductPydantic(name="Smartwatch", sku="SW-005", price=299.99, inventory=12)
]

# Batch convert to OGM models (all in a single transaction)
product_ogms = Converter.batch_to_ogm(products)

print(f"Converted {len(product_ogms)} products to OGM models")

# Batch convert back to Pydantic models
products_pydantic = Converter.batch_to_pydantic(product_ogms)

for product in products_pydantic:
  print(product.model_dump())
```

Output:

``` 
Converted 5 products to OGM models
{'name': 'Laptop', 'sku': 'LT-001', 'price': 1299.99, 'inventory': 10}
{'name': 'Smartphone', 'sku': 'SP-002', 'price': 899.99, 'inventory': 15}
{'name': 'Headphones', 'sku': 'HP-003', 'price': 199.99, 'inventory': 25}
{'name': 'Tablet', 'sku': 'TB-004', 'price': 499.99, 'inventory': 8}
{'name': 'Smartwatch', 'sku': 'SW-005', 'price': 299.99, 'inventory': 12}
```

</details>

<details>
<summary>Dictionary Conversion</summary>

This example shows conversions between dictionaries and OGM models:

```python
from neomodel import StructuredNode, StringProperty, IntegerProperty, config, RelationshipTo

from pydantic_neomodel_dict import Converter

# Set up Neomodel (Neo4j) connection - this is required!
config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'
config.ENCRYPTED_CONNECTION = False
config.AUTO_INSTALL_LABELS = True


# Define Neomodel (Neo4j) OGM models
class AddressOGM(StructuredNode):
  street = StringProperty(required=True)
  city = StringProperty(required=True)
  zip_code = StringProperty(required=True)


class PersonOGM(StructuredNode):
  name = StringProperty(required=True)
  age = IntegerProperty(required=True)
  address = RelationshipTo(AddressOGM, 'LIVES_AT')


# Dictionary data with nested relationship
person_dict = {
  "name": "Alex Johnson",
  "age": 32,
  "address": {
    "street": "456 Oak Avenue",
    "city": "San Francisco",
    "zip_code": "94102"
  }
}

# Convert dictionary to OGM model
person_ogm = Converter.dict_to_ogm(person_dict, PersonOGM)

# Convert OGM model back to dictionary
person_dict_again = Converter.ogm_to_dict(person_ogm)

print(person_dict)
print(person_dict_again)
print(f"Person: {person_dict_again['name']}, Age: {person_dict_again['age']}")
print(f"Address: {person_dict_again['address']['street']}, {person_dict_again['address']['city']}")
```

Output:

``` 
{'name': 'Alex Johnson', 'age': 32, 'address': {'street': '456 Oak Avenue', 'city': 'San Francisco', 'zip_code': '94102'}}
{'name': 'Alex Johnson', 'age': 32, 'address': {'street': '456 Oak Avenue', 'city': 'San Francisco', 'zip_code': '94102'}}
Person: Alex Johnson, Age: 32
Address: 456 Oak Avenue, San Francisco
```

</details>

## API Reference

### Core Methods

- `Converter.register_models(pydantic_class, ogm_class)`: Register mapping between Pydantic and OGM models
- `Converter.to_ogm(pydantic_instance, ogm_class=None, max_depth=10)`: Convert Pydantic instance to OGM
- `Converter.to_pydantic(ogm_instance, pydantic_class=None, max_depth=10)`: Convert OGM instance to Pydantic
- `Converter.dict_to_ogm(data_dict, ogm_class, max_depth=10)`: Convert dictionary to OGM instance
- `Converter.ogm_to_dict(ogm_instance, max_depth=10)`: Convert OGM instance to dictionary

### Batch Operations

- `Converter.batch_to_ogm(pydantic_instances, ogm_class=None, max_depth=10)`: Convert multiple Pydantic instances to OGM
- `Converter.batch_to_pydantic(ogm_instances, pydantic_class=None, max_depth=10)`: Convert multiple OGM instances to
  Pydantic
- `Converter.batch_dict_to_ogm(data_dicts, ogm_class, max_depth=10)`: Convert multiple dictionaries to OGM instances
- `Converter.batch_ogm_to_dict(ogm_instances, max_depth=10)`: Convert multiple OGM instances to dictionaries

### Custom Type Conversion

- `Converter.register_type_converter(source_type, target_type, converter_func)`: Register custom type converter function

## Limitations

- **Default Neomodel (Neo4j) Connection**: This library uses the default `db` connection from `neomodel`, so creating OGM models
  not in global scope may lead to errors. Always ensure your Neomodel (Neo4j) connection is properly configured before using the
  converter.
- **Depth Limit**: Conversion has a default depth limit of 10 to prevent excessive recursion in complex object graphs.
- **Transaction Management**: The converter handles transactions internally but doesn't provide explicit transaction
  control features.
- **Performance**: Converting very large object graphs may impact performance, especially with deep nesting levels.
- **Pydantic Versions**: Currently supports Pydantic 2.0.0+; compatibility with older versions is not guaranteed.
- **Node Identity**: The converter uses object identity for cycle detection, which may not work correctly in all edge
  cases.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see
the [LICENSE](https://github.com/HardMax71/pydantic-neomodel-dict/blob/main/LICENSE) file for details.