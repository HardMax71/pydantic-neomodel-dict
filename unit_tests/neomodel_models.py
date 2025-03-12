"""
Models Module

This module defines all neomodel StructuredNode classes used in the application.
By centralizing all model definitions in a single file, we ensure that each model is
defined exactly once per Python process. This approach avoids duplicate registration
in neomodel's global registry (db._NODE_CLASS_REGISTRY) and prevents errors such as
neomodel.exceptions.NodeClassAlreadyDefined.

For further information, please refer to:
    • Neomodel Documentation on Extending neomodel:
      https://neomodel.readthedocs.io/en/latest/extending.html
    • Related discussions on Stack Overflow:
      https://stackoverflow.com/questions/64851360/neomodel-class-definition
    • Neo4j Developer Documentation:
      https://neo4j.com/developer/

"""


