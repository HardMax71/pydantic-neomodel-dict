from neomodel import IntegerProperty, One, RelationshipFrom, RelationshipTo, StringProperty, StructuredNode, db
from pydantic import BaseModel

from pydantic_neomodel_dict import Converter


# Define OGM models
class Address2OGM(StructuredNode):
    street = StringProperty(required=True)
    city = StringProperty(required=True)
    zip_code = StringProperty(required=True)


class Order2OGM(StructuredNode):
    order_id = StringProperty(unique_index=True, required=True)
    amount = IntegerProperty(required=True)
    # Many-to-one relationship
    customer = RelationshipFrom('Customer2OGM', 'PLACED')


class Customer2OGM(StructuredNode):
    name = StringProperty(required=True)
    email = StringProperty(unique_index=True, required=True)
    # One-to-one relationship
    address = RelationshipTo(Address2OGM, 'HAS_ADDRESS', cardinality=One)
    # One-to-many relationship
    orders = RelationshipTo(Order2OGM, 'PLACED')


# Define Pydantic models
class AddressModel(BaseModel):
    street: str
    city: str
    zip_code: str


class OrderModel(BaseModel):
    order_id: str
    amount: int


class CustomerModel(BaseModel):
    name: str
    email: str
    address: AddressModel
    orders: list[OrderModel] = []


class TestDictConversion:
    """Test automatic relationship unwrapping behavior in ogm_to_dict method"""

    def setup_method(self):
        """Setup test database"""
        # Clear the database before each test
        db.cypher_query("MATCH (n) DETACH DELETE n")

        # Register models
        Converter.register_models(AddressModel, Address2OGM)
        Converter.register_models(OrderModel, Order2OGM)
        Converter.register_models(CustomerModel, Customer2OGM)

    def teardown_method(self):
        """Clean up after test"""
        # Clear the database after each test
        db.cypher_query("MATCH (n) DETACH DELETE n")

        # Clear registrations
        Converter._pydantic_to_ogm = {}
        Converter._ogm_to_pydantic = {}

    def test_one_to_one_relationship_conversion(self, db_connection):
        """Test that one-to-one relationships are properly represented"""
        # Create test data
        person_dict = {
            "name": "John Smith",
            "email": "john@example.com",
            "address": {
                "street": "123 Main St",
                "city": "Boston",
                "zip_code": "02108"
            }
        }

        # Convert dictionary to OGM
        customer_ogm = Converter.dict_to_ogm(person_dict, Customer2OGM)

        # Convert OGM back to dictionary
        result_dict = Converter.ogm_to_dict(customer_ogm)

        # Verify one-to-one relationship is represented as dict
        assert "address" in result_dict
        assert isinstance(result_dict["address"], dict)
        assert result_dict["address"]["street"] == "123 Main St"
        assert result_dict["address"]["city"] == "Boston"

        # Verify we can access nested fields directly
        street = result_dict["address"]["street"]
        assert street == "123 Main St"

    def test_one_to_many_relationship_multiple_items(self, db_connection):
        """Test that one-to-many relationships with multiple items remain as lists"""
        # Create customer with multiple orders
        customer_dict = {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "address": {
                "street": "456 Oak Ave",
                "city": "Chicago",
                "zip_code": "60601"
            },
            "orders": [
                {"order_id": "ORD-001", "amount": 99},
                {"order_id": "ORD-002", "amount": 45}
            ]
        }

        # Convert dictionary to OGM
        customer_ogm = Converter.dict_to_ogm(customer_dict, Customer2OGM)

        # Convert OGM back to dictionary
        result_dict = Converter.ogm_to_dict(customer_ogm)

        # Verify orders is a list since it contains multiple items
        assert "orders" in result_dict
        assert isinstance(result_dict["orders"], list)
        assert len(result_dict["orders"]) == 2

        # Verify order details
        order_ids = sorted([order["order_id"] for order in result_dict["orders"]])
        assert order_ids == ["ORD-001", "ORD-002"]

    def test_one_to_many_relationship_single_item(self, db_connection):
        """Test that one-to-many relationships with a single item gets unwrapped to a dict"""
        # Create customer with only one order
        customer_dict = {
            "name": "Jane Smith",
            "email": "jsmith@example.com",
            "address": {
                "street": "456 Oak Ave",
                "city": "Chicago",
                "zip_code": "60601"
            },
            "orders": [
                {"order_id": "ORD-001", "amount": 99}
            ]  # Only one order
        }

        # Convert dictionary to OGM
        customer_ogm = Converter.dict_to_ogm(customer_dict, Customer2OGM)

        # Convert OGM back to dictionary
        result_dict = Converter.ogm_to_dict(customer_ogm)

        # Verify single order was NOT unwrapped to a dict
        assert "orders" in result_dict
        assert isinstance(result_dict["orders"], list)
        assert result_dict["orders"][0]["order_id"] == "ORD-001"
        assert result_dict["orders"][0]["amount"] == 99

    def test_empty_relationships(self, db_connection):
        """Test conversion of empty relationships"""
        # Create customer with no orders and no address
        customer = Customer2OGM(name="No Relations", email="empty@example.com").save()

        # Convert OGM to dictionary
        result_dict = Converter.ogm_to_dict(customer)

        # Empty relationship results
        # Expecting empty list due to adding key of Customer2OGM, CustomerModel,
        #  latter has default of "orders" = [] and for address = None
        assert "address" in result_dict
        assert result_dict["address"] is None
        assert "orders" in result_dict
        assert result_dict["orders"] == []

    def test_nested_dict_structures(self, db_connection):
        """Test nested dictionaries with multiple levels of relationships"""
        # Create a customer with address and orders
        # where orders have customers (circular reference)
        customer_dict = {
            "name": "Complex Customer",
            "email": "complex@example.com",
            "address": {
                "street": "Complex Street",
                "city": "Complex City",
                "zip_code": "12345"
            },
            "orders": [
                {"order_id": "COMPLEX-1", "amount": 123},
                {"order_id": "COMPLEX-2", "amount": 456}
            ]
        }

        # Convert dictionary to OGM
        customer_ogm = Converter.dict_to_ogm(customer_dict, Customer2OGM)

        # Get orders and connect them back to the customer (circular reference)
        orders = list(customer_ogm.orders.all())
        for order in orders:
            order.customer.connect(customer_ogm)

        # Convert OGM back to dictionary
        result_dict = Converter.ogm_to_dict(customer_ogm)

        # Verify structure - should have orders as a list
        assert isinstance(result_dict["orders"], list)

        # Each order should reference back to the customer
        for order in result_dict["orders"]:
            assert "customer" in order
            assert order["customer"]["email"] == "complex@example.com"

    def test_batch_ogm_to_dict(self, db_connection):
        """Test batch_ogm_to_dict with various relationship types"""
        # Create multiple customers with different relationship structures
        customer1 = Customer2OGM(name="Customer One", email="one@example.com").save()
        addr1 = Address2OGM(street="Street 1", city="City 1", zip_code="11111").save()
        customer1.address.connect(addr1)

        # Customer with two orders
        customer2 = Customer2OGM(name="Customer Two", email="two@example.com").save()
        order1 = Order2OGM(order_id="ORD-1", amount=100).save()
        order2 = Order2OGM(order_id="ORD-2", amount=200).save()
        customer2.orders.connect(order1)
        customer2.orders.connect(order2)

        # Customer with one order
        customer3 = Customer2OGM(name="Customer Three", email="three@example.com").save()
        order3 = Order2OGM(order_id="ORD-3", amount=300).save()
        customer3.orders.connect(order3)

        # Convert batch with different relationship structures
        results = Converter.batch_ogm_to_dict([customer1, customer2, customer3])

        # Verify results
        assert len(results) == 3

        # Customer 1: One-to-one relationship
        customer1_dict = next(r for r in results if r["email"] == "one@example.com")
        assert isinstance(customer1_dict["address"], dict)

        # Customer 2: One-to-many relationship with multiple items
        customer2_dict = next(r for r in results if r["email"] == "two@example.com")
        assert isinstance(customer2_dict["orders"], list)
        assert len(customer2_dict["orders"]) == 2

        # Customer 3: One-to-many relationship with single item
        # (should NOT be unwrapped due to typehint in Pydantic model)
        customer3_dict = next(r for r in results if r["email"] == "three@example.com")
        assert isinstance(customer3_dict["orders"], list)
        assert customer3_dict["orders"][0]["order_id"] == "ORD-3"
