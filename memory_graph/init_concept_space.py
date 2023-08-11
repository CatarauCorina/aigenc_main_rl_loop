import os
from neomodel import config
from neomodel import (StructuredNode, StringProperty, IntegerProperty,
    UniqueIdProperty, RelationshipTo)


config.DATABASE_URL = os.environ["NEO4J_BOLT_URL"]
config.NEO_USER = os.environ['NEO_USER']
config.NEO_PASS = os.environ['NEO_PASS']


config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'


class ObjectConcept(StructuredNode):
    value = StringProperty(required=True)
    connections = RelationshipTo('ObjectConcept', 'ACTION')
