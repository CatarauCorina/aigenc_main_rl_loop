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

    def add_nodes_to_state(self, obj_tensors, state_id):
        tensor_list = []
        for obj in obj_tensors.squeeze(0).squeeze(0):
            obj_list = obj.tolist()
            obj_id = self.add_data('ObjectConcept')
            self.update_node_by_id(obj_id['elementId(n)'][0], obj_list)
            self.match_state_add_node(state_id, obj_id['elementId(n)'][0])
        return tensor_list

    def add_state_with_objects(self, state_encoding, objects, episode_id, time):
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (ep: Episode) where elementId(ep) = "{episode_id}"
                CREATE
                    (st:StateT {{state_enc: {state_encoding.tolist()[0]} }}) 
                CREATE (ep)-[:`has_state`{{t:{time} }}]->(st) return elementId(st)
                """
            )
            result_state_creation = pd.DataFrame([r.values() for r in result], columns=result.keys())
        state_id = result_state_creation['elementId(st)'][0]
        self.add_nodes_to_state(objects, state_id)
        return

    def match_state_add_node(self, state_id, node_id):
        with self.driver.session() as session:
            result = session.run(
                f"""MATCH (s:StateT), (n:ObjectConcept) WHERE elementId(s) = "{state_id}" and elementId(n) ="{node_id}"
                CREATE (s)-[:`has_object`]->(n)
                """
            )
            return result

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