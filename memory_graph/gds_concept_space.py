import os
import pandas as pd
from neo4j import GraphDatabase


class ConceptSpaceGDS:
    DATABASE_URL = os.environ["NEO4J_BOLT_URL"]
    NEO_USER = os.environ['NEO_USER']
    NEO_PASS = os.environ['NEO_PASS']

    def __init__(self, memory_type='workingMemory'):
        self.driver = GraphDatabase.driver(self.DATABASE_URL, auth=(self.NEO_USER, self.NEO_PASS))
        self.memory_type = memory_type
        self.set_memory()

    def close(self):
        self.driver.close()

    def add_data(self, node_type):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    CREATE (n:{node_type}) return elementId(n)
                """
            )
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def set_property(self, object_id, node_type,property, property_value, is_string=False):
        query_string = f"""
                           USE {self.memory_type}
                           MATCH (n:{node_type}) where elementId(n)="{object_id}"
                           SET n.{property}={property_value}
                       """
        if is_string:
            query_string = f"""
                           USE {self.memory_type}
                           MATCH (n:{node_type}) where elementId(n)="{object_id}"
                           SET n.{property}="{property_value}"
                       """
        with self.driver.session() as session:
            result = session.run(query_string)
            return

    def set_memory(self):
        with self.driver.session() as session:
            result = session.run(
                f"""USE {self.memory_type} RETURN null"""
            )
            return

    def add_nodes_to_state(self, obj_tensors, state_id):
        tensor_list = []
        for obj in obj_tensors.squeeze(0).squeeze(0):
            obj_list = obj.tolist()
            obj_id = self.add_data('ObjectConcept')
            self.update_node_by_id(obj_id['elementId(n)'][0], obj_list)
            self.match_state_add_node(state_id, obj_id['elementId(n)'][0])
        return tensor_list

    def add_state_with_objects(self, state_encoding, episode_id, time):
        with self.driver.session() as session:
            result = session.run(
                f"""
                USE {self.memory_type}
                MATCH (ep: Episode) where elementId(ep) = "{episode_id}"
                CREATE
                    (st:StateT {{
                                state_enc: {state_encoding.tolist()[0]}, 
                                episode_id:toInteger(split("{episode_id}",":")[2])
                            }}
                    ) 
                CREATE (ep)-[:`has_state`{{t:{time} }}]->(st) return elementId(st)
                """
            )
            result_state_creation = pd.DataFrame([r.values() for r in result], columns=result.keys())
        state_id = result_state_creation['elementId(st)'][0]
        return state_id

    def match_state_add_node(self, state_id, node_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                USE {self.memory_type}
                MATCH (s:StateT), (n:ObjectConcept) WHERE elementId(s) = "{state_id}" and elementId(n) ="{node_id}"
                MERGE (s)-[:`has_object`]->(n)
                """
            )
            return result

    def match_obj_add_action(self, obj_id, action_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                            USE {self.memory_type}
                            MATCH (o:ObjectConcept), (a:ActionRepr) WHERE elementId(o) = "{obj_id}" and elementId(a) ="{action_id}"
                            MERGE (o)-[:`contribute`]->(a)
                            """
            )
            return result

    def match_state_add_aff(self, state_id, node_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                 USE {self.memory_type}
                 MATCH (s:StateT), (n:Affordance) WHERE elementId(s) = "{state_id}" and elementId(n) ="{node_id}"
                 MERGE (s)-[:`influences`]->(n)
                 """
            )
            return result

    def match_state_add_aff_outcome(self, state_id, node_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    MATCH (s:StateT), (n:Affordance) WHERE elementId(s) = "{state_id}" and elementId(n) ="{node_id}"
                    MERGE (n)-[:`outcome`]->(s)
                    """
            )
            return result


    def match_action_add_aff(self, action_id, node_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                 USE {self.memory_type}
                 MATCH (act:ActionRepr), (n:Affordance) WHERE elementId(act) = "{action_id}" and elementId(n) ="{node_id}"
                 MERGE (act)-[:`produces`]->(n)
                 """
            )
            return result




    def update_node_by_id(self, node_id, value):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    MATCH (s) WHERE elementId(s) = "{node_id}" set s.value={value}
                """
            )
            return result

    def get_episode_graph(self, episode_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    Match (e:Episode)-[:has_state]->(s:StateT)-[:has_object]->(o:ObjectConcept) where elementId(e)={episode_id} return e,s,o
                """
            )
            return result


    def add_data_props(self, node_type, props):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    CREATE (n:{node_type} ${props})
                """
            )
            return result

    def fetch_data(self, query, params={}):
        with self.driver.session() as session:
            result = session.run(query, params)
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def objects_attention_and_reinforcer(self, time, episode_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    match (e:Episode)-[r:has_state]->(s:StateT)-[r1:has_object]->(o:ObjectConcept)
                    where s.episode_id={episode_id} and r.t={time} 
                    return s.reward,r.t,s.reward-sum(o.att) as reinforcer, sum(o.att)
            
                 """
            )
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def get_obj_att_at_time_t(self, time, episode_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    match (e:Episode)-[r:has_state]->(s:StateT)-[r1:has_object]->(o:ObjectConcept) 
                    where s.episode_id={episode_id} and r.t={time}
                    return elementId(o) as id_o , o.att,o.alpha as alpha, o.all_att_values as att_values
                """
            )
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def get_obj_att_values_prev_time(self, time, episode_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                     USE {self.memory_type}
                    match (e:Episode)-[r:has_state]->(s:StateT)-[r1:has_object]->(o:ObjectConcept) 
                    where s.episode_id={episode_id} and r.t < {time}
                    return elementId(o) as id_o ,collect(r.t),o.att as prev_att,o.alpha as alpha, collect(s.reward)
                """
            )
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def update_objects_attention(self, object_ids_att_values):
        with self.driver.session() as session:
            # result = session.run(
            #     f"""
            #         USE {self.memory_type}
            #         UNWIND {object_ids_att_values} AS p
            #         MATCH (o:ObjectConcept) WHERE elementId(o) = p.elementId(o)
            #         SET o.att = p.new_value_obj_i
            #     """
            # )
            r = session.run(
                f"USE {self.memory_type}\
                UNWIND $obj_batch as obj \
                MATCH (o:ObjectConcept) WHERE elementId(o) = obj.id_o \
                SET o.att = obj.new_value_obj_i set o.all_att_values=obj.att_values set o.alpha=obj.alpha_obj_i", obj_batch=object_ids_att_values
            )
            return


    def get_attention_for_episode(self, ep_id="4:35c6f93f-aedf-40a2-ba92-c88bc937e420:883"):
        with self.driver.session() as session:
            result = session.run(
                f"""USE {self.memory_type}\
                      match(e:Episode)-[r:has_state]-(s)-[:has_object]->(o) where elementId(e)="{ep_id}"\
                      return elementId(o),collect(elementId(s)), collect(s.reward),collect(r.t),collect(o.all_att_values), o.alpha"""
            )
            return pd.DataFrame([r.values() for r in result], columns=result.keys())


    def get_objects_associated_with_reward(self, reward_value=0.01):
        with self.driver.session() as session:
            result = session.run(
                f"""USE {self.memory_type}\
                      match (s:StateT)-[:has_object]->(o:ObjectConcept) where s.reward >= {reward_value} return elementId(o) as obj_id, collect(o.all_att_values) as obj_values"""
            )
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def get_objects_associated_with_0reward(self, reward_value=0):
        with self.driver.session() as session:
            result = session.run(
                f"""USE {self.memory_type}\
                      match (s:StateT)-[:has_object]->(o:ObjectConcept) where s.reward={reward_value} return elementId(o) as obj_id, collect(o.all_att_values) as obj_values"""
            )
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

if __name__ == "__main__":
    cs_memory = ConceptSpaceGDS()
    cs_memory.fetch_data("""
    CALL db.propertyKeys
    """)