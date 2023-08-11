import os
import pandas as pd
from neo4j import GraphDatabase


class ConceptSpaceGDS:
    DATABASE_URL = os.environ["NEO4J_BOLT_URL"]
    NEO_USER = os.environ['NEO_USER']
    NEO_PASS = os.environ['NEO_PASS']

    def __init__(self):
        self.driver = GraphDatabase.driver(self.DATABASE_URL, auth=(self.NEO_USER, self.NEO_PASS))

    def close(self):
        self.driver.close()

    def add_data(self, node_type):
        with self.driver.session() as session:
            result = session.run(
                f"""CREATE (n:{node_type}) return elementId(n)"""
            )
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def update_node_by_id(self, node_id, value):
        with self.driver.session() as session:
            result = session.run(
                f"""MATCH (s) WHERE elementId(s) = "{node_id}" set s.value={value}"""
            )
            return result

    def add_data_props(self, node_type, props):
        with self.driver.session() as session:
            result = session.run(
                f"""CREATE (n:{node_type} ${props})"""
            )
            return result

    def fetch_data(self, query, params={}):
        with self.driver.session() as session:
            result = session.run(query, params)
            return pd.DataFrame([r.values() for r in result], columns=result.keys())


if __name__ == "__main__":
    cs_memory = ConceptSpaceGDS()
    cs_memory.fetch_data("""
    CALL db.propertyKeys
    """)