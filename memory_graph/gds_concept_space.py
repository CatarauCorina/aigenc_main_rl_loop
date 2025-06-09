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

    def minigrid_add_episode(self, success, task):
        with self.driver.session() as session:
            result = session.run(
                f"USE {self.memory_type} CREATE (e:Episode {{succesfull_outcome: $success, task: $task}}) RETURN elementId(e) AS eid",
                 success=success, task=task
            )
            return result.single()["eid"]

    def minigrid_add_state(self, episode_id, timestep, embedding):
        with self.driver.session() as session:
            result = session.run(
                f"""
                USE {self.memory_type}
                MATCH (e:Episode) WHERE elementId(e) = $eid
                CREATE (s:StateT {{state_enc: $emb}})
                CREATE (e)-[:has_state {{t: $timestep}}]->(s)
                RETURN elementId(s) AS sid
                """, eid=episode_id, timestep=timestep, emb=embedding
            )
            return result.single()["sid"]


    def minigrid_add_object_concept(self, state_id, emb, label):
        with self.driver.session() as session:
            result = session.run(
                f"""
                USE {self.memory_type}
                MATCH (s:StateT) WHERE elementId(s) = $sid
                CREATE (o:ObjectConcept {{value: $embedding, name: $label}})
                CREATE (s)-[:has_object]->(o)
                RETURN elementId(o) AS oid
                """, sid=state_id, embedding=emb, label=label
            )
            return result.single()["oid"]

    def minigrid_add_edge(self, source, target, label, weight=1.0, state_id=None, source_tag=None):
        """
        Adds or updates a labeled edge (relationship) between two nodes in the graph, with metadata.

        Args:
            source (str): Element ID of the source node (e.g., ObjectConcept).
            target (str): Element ID of the target node (e.g., State).
            label (str): Label/type of the relationship (e.g., "contribute").
            weight (float): Optional weight value for the edge.
            state_id (str): Optional reference to the state that generated this link.
            source_tag (str): Optional tag indicating the origin of the link (e.g., "setle_enrich").
        """
        with self.driver.session() as session:
            session.run(
                f"""
                   USE {self.memory_type}
                   MATCH (a) WHERE elementId(a) = $source
                   MATCH (b) WHERE elementId(b) = $target
                   MERGE (a)-[r:{label}]->(b)
                   SET r.weight = $weight,
                       r.state_id = $state_id,
                       r.source_tag = $source_tag
                   """,
                source=source,
                target=target,
                weight=weight,
                state_id=state_id,
                source_tag=source_tag
            )

    def minigrid_create_node(self, label, props, source="manual"):
        """
        Create a node with the given label and properties in Neo4j.
        Args:
            label (str): Node label (e.g., "Affordance", "ObjectConcept")
            props (dict): Dictionary of properties to assign to the node
            source (str): Optional string tag for provenance
        Returns:
            str: Neo4j elementId of the created node
        """
        with self.driver.session() as session:
            query = f"""
               USE {self.memory_type}
               CREATE (n:{label})
               SET n += $props, n.source = $source
               RETURN elementId(n) AS node_id
               """
            result = session.run(query, props=props, source=source)
            return result.single()["node_id"]

    def add_data(self, node_type):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    CREATE (n:{node_type}) return elementId(n)
                """
            )
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def clear_wm(self):
        with self.driver.session() as session:
            result = session.run(
                f"""
                              USE {self.memory_type}
                              MATCH (n) detach delete n
                              """
            )
            return

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

    def get_property(self, node_type, object_id, property):
        query_string = f"""
                                  USE {self.memory_type}
                                  MATCH (n:{node_type}) where elementId(n)="{object_id}"
                                  return n.{property}
                              """
        with self.driver.session() as session:
            result = session.run(query_string)
            return result.data()

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

    def create_node(self, label: str, properties: dict, source: str = "manual"):
        """
        Creates a new node with a given label and properties.

        Parameters:
        - label (str): The Neo4j label for the node (e.g., "Affordance", "ObjectConcept").
        - properties (dict): Properties to assign to the node.
        - source (str): Optional source tag to trace the node's origin (default: "manual").

        Returns:
        - str: The element ID of the newly created node.
        """
        with self.driver.session() as session:
            # Ensure the 'source' metadata is added
            properties['source'] = source

            prop_str = ", ".join([f"{k}: ${k}" for k in properties])
            query = f"""
            USE {self.memory_type}
            CREATE (n:`{label}` {{ {prop_str} }})
            RETURN elementId(n) AS node_id
            """
            result = session.run(query, properties)
            record = result.single()
            return record["node_id"] if record else None

    def match_state_add_generic_node(self, state_id, node_id, node_label, rel_type="enriched_with"):
        """
        Create a relationship between a StateT node and another node type (ObjectConcept, ActionRepr, etc.)

        Args:
            state_id (str): Neo4j ID of the state node.
            node_id (str): Neo4j ID of the node to attach.
            node_label (str): Label of the target node (e.g., "ObjectConcept").
            rel_type (str): Type of relationship to create (default: "enriched_with").
        """
        with self.driver.session() as session:
            result = session.run(
                f"""
                USE {self.memory_type}
                MATCH (s:StateT), (n:{node_label})
                WHERE elementId(s) = $state_id AND elementId(n) = $node_id
                MERGE (s)-[:`{rel_type}`]->(n)
                """,
                state_id=state_id,
                node_id=node_id
            )
            return result

    def match_state_add_encs(self, state_id, z_sc, z_mp):
        with self.driver.session() as session:
            result = session.run(
                f"""
                USE {self.memory_type}
                MATCH (s:StateT) WHERE elementId(s) = "{state_id}"
                SET s.zsc="{z_sc}"
                SET s.zmp="{z_mp}"
                return s
                """
            )
            return result

    def get_connected_nodes(self, node_id, edge_type=None):
        with self.driver.session() as session:
            if edge_type:
                query = f"""
                USE {self.memory_type}
                MATCH (a)-[r:`{edge_type}`]-(b)
                WHERE elementId(a) = $node_id
                RETURN b
                """
            else:
                query = f"""
                USE {self.memory_type}
                MATCH (a)--(b)
                WHERE elementId(a) = $node_id
                RETURN b
                """
            result = session.run(query, node_id=node_id)
            return [record['b'] for record in result]

    def add_edge(self, source, target, label, weight=1.0, state_id=None, source_tag="manual"):
        """
        Add an edge between two nodes with optional weight and tagging.

        Parameters:
        - source (str): elementId of the source node.
        - target (str): elementId of the target node.
        - label (str): relationship type (e.g., 'has_object', 'interacts_with').
        - weight (float): edge weight (default = 1.0).
        - state_id (str): optional context state (used to tag enrichment location).
        - source_tag (str): tag describing source of enrichment (e.g., 'setle').
        """
        with self.driver.session() as session:
            query = f"""
            USE {self.memory_type}
            MATCH (a), (b)
            WHERE elementId(a) = $source AND elementId(b) = $target
            MERGE (a)-[r:`{label}`]->(b)
            SET r.weight = $weight,
                r.source = $source_tag
            """
            session.run(query, {
                "source": source,
                "target": target,
                "weight": weight,
                "source_tag": source_tag
            })

    def get_ids_node(self, node_name='ObjectConcept'):
        query = f"""
                USE {self.memory_type}
                match (o:{node_name}) return elementId(o)"""
        with self.driver.session() as session:
            result = session.run(query)
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def get_all_object_concepts(self, use_label=False):
        with self.driver.session() as session:
            if use_label:
                result = session.run(
                    f"""
                                USE {self.memory_type}
                                MATCH (o:ObjectConcept)
                                RETURN elementId(o) AS id, o.value AS embedding, o.label AS label
                                """
                )
                return [(record["id"], record["embedding"], record['label']) for record in result]

            result = session.run(
                f"""
                USE {self.memory_type}
                MATCH (o:ObjectConcept)
                RETURN elementId(o) AS id, o.value AS embedding
                """
            )
            return [(record["id"], record["embedding"]) for record in result]

    def find_similar_object_concepts(self, candidate_embedding, threshold=0.9, use_label=False, check_label=None, fetch_obj=[]):
        import torch
        import torch.nn.functional as F
        """
        Find ObjectConcept nodes in the working memory that are similar to the given embedding.

        Args:
            candidate_embedding (torch.Tensor): The embedding to compare against existing WM nodes.
            threshold (float): Cosine similarity threshold to consider nodes as similar.

        Returns:
            List[str]: A list of matching node IDs in working memory.
        """
        similar_nodes = []
        if len(fetch_obj) == 0:
            all_objects = self.get_all_object_concepts(use_label=use_label)
        else:
            all_objects = fetch_obj
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        candidate_embedding = torch.tensor(candidate_embedding).to(device)
        if use_label:
            for id, obj_emb, label in all_objects:
                existing_emb = torch.tensor(obj_emb, dtype=torch.float32).to(device)
                similarity = F.cosine_similarity(candidate_embedding.unsqueeze(0), existing_emb.unsqueeze(0)).item()
                if similarity >= threshold and check_label==label:
                    similar_nodes.append(id)
        else:
            if len(all_objects) == 0:
                return [], all_objects
            ids, embeddings = zip(*all_objects)  # unzip the IDs and embeddings

            # Convert all to a tensor
            existing_embs = torch.tensor(embeddings, dtype=torch.float32).to(device)  # shape: [N, D]
            candidate_emb = candidate_embedding.to(device).unsqueeze(0)  # shape: [1, D]

            # Compute cosine similarity between candidate and all existing embeddings
            similarities = F.cosine_similarity(candidate_emb, existing_embs)  # shape: [N]

            # Get IDs where similarity is above threshold
            similar_nodes = [id_ for id_, sim in zip(ids, similarities) if sim.item() >= threshold]

            # for id,obj_emb in all_objects:
            #     existing_emb = torch.tensor(obj_emb, dtype=torch.float32).to(device)
            #     similarity = F.cosine_similarity(candidate_embedding.unsqueeze(0), existing_emb.unsqueeze(0)).item()
            #     if similarity >= threshold:
            #         similar_nodes.append(id)

        return similar_nodes, all_objects

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


    def get_state_graph(self, state_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    match (n:StateT)-[r1:has_object]->(o:ObjectConcept)
                    where elementId(n)='{state_id}'
                    with n,r1,o
                    optional match (o)-[r2:contribute]->(a:ActionRepr)
                    optional match (n)-[r3:influences]->(aff:Affordance) return n,o,a, aff,r1,r2,r3
                """
            )
            nodes = list(result.graph()._nodes.values())
            rels = list(result.graph()._relationships.values())

            return nodes, rels

    def get_state_graph_2(self, state_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    
                    match (s:StateT) where elementId(s)="{state_id}" with s
                    match (s)-[r:has_object]->(o:ObjectConcept) with s,o,r
                    optional match (s)-[r1:influences]->(aff:Affordance) with s,o,aff,r,r1
                    MATCH (s)-[r2:has_object]->(o1:ObjectConcept)-[r3:contribute]->(a:ActionRepr)
                    MATCH (s)-[r4:has_object]->(o2:ObjectConcept)-[r5:contribute]->(a)
                    WHERE id(o1) <> id(o2) AND id(o) <> id(o1) and id(o) <> id(o2)
                    RETURN s, o1, o2, a, o, aff,r,r1,r2,r3,r4,r5
                """
            )
            nodes = list(result.graph()._nodes.values())
            rels = list(result.graph()._relationships.values())

            return nodes, rels

    def get_state_graph_setle(self, state_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                USE {self.memory_type}
                MATCH (s:StateT) WHERE elementId(s) = '{state_id}'
                OPTIONAL MATCH (ep:Episode)-[r:has_state]->(s)
                OPTIONAL MATCH (s)-[r1:has_object]->(o:ObjectConcept)
                OPTIONAL MATCH (s)-[r2:produces]->(aff:Affordance)
                OPTIONAL MATCH (aff)-[r3:outcome]->(o2:ObjectConcept)
                OPTIONAL MATCH (o)-[r4:contribute]->(a:ActionRepr)
                OPTIONAL MATCH (a)-[r5:contribute]->(o3:ObjectConcept)
                RETURN ep, s, o, aff, o2, a, o3, r,r1, r2, r3, r4, r5
                """
            )
            nodes = list(result.graph()._nodes.values())
            rels = list(result.graph()._relationships.values())
            return nodes, rels

    def get_state_graph_setle_minigrid(self, state_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                 USE {self.memory_type}
                 MATCH (s:StateT) WHERE elementId(s) = '{state_id}'
                 OPTIONAL MATCH (ep:Episode)-[r:has_state]->(s)
                 OPTIONAL MATCH (s)-[r1:has_object]->(o:ObjectConcept)
                 OPTIONAL MATCH (s)-[r2:produces]->(aff:Affordance)
                 OPTIONAL MATCH (aff)-[r3:outcome]->(o2:ObjectConcept)
                 RETURN ep, s, o, aff, o2, r,r1, r2, r3
                 """
            )
            nodes = list(result.graph()._nodes.values())
            rels = list(result.graph()._relationships.values())
            return nodes, rels

    def get_all_episode_embeddings(self):
        import torch
        """
        Fetches all episodes with a stored 'set_embedding' from the Neo4j LTM.

        Returns:
            Dict[str, torch.Tensor]: {episode_id: embedding_tensor}
        """
        query = f"""
        USE {self.memory_type}
         MATCH (e:Episode)
        WHERE e.set_embedding is not null
        RETURN elementId(e) as eid, e.set_embedding AS embedding, e.task as task
        """

        with self.driver.session() as session:
            results = session.run(query)
            return {
                row['eid']: torch.tensor(row['embedding'], dtype=torch.float32).cuda()
                for row in results if row['embedding'] is not None
            }

    def get_all_episode_tasks(self):
        import torch
        """
        Fetches all episodes with a stored 'set_embedding' from the Neo4j LTM.

        Returns:
            Dict[str, torch.Tensor]: {episode_id: embedding_tensor}
        """
        query = f"""
        USE {self.memory_type}
         MATCH (e:Episode)
        WHERE e.set_embedding is not null
        RETURN elementId(e) as eid, e.set_embedding AS embedding, e.task as task
        """

        with self.driver.session() as session:
            results = session.run(query)
            return  {
                row['eid']: row['task']
                for row in results if row['embedding'] is not None
            }


    def get_crt_episode_task(self,ep_id):
        """
        Fetches all episodes with a stored 'set_embedding' from the Neo4j LTM.

        Returns:
            Dict[str, torch.Tensor]: {episode_id: embedding_tensor}
        """
        query = f"""
        USE {self.memory_type}
         MATCH (e:Episode)
        WHERE elementId(e)="{ep_id}"
        RETURN elementId(e) as eid, e.set_embedding AS embedding, e.task as task
        """

        with self.driver.session() as session:
            results = session.run(query)
            return  {
                row['eid']: row['task']
                for row in results if row['embedding'] is not None
            }


    def get_episode_graph(self, episode_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}

                     match (e:Episode)-[r:has_state]->(s:StateT)-[r1:has_object]->(o:ObjectConcept)
                     where elementId(e)="{episode_id}"
                     with e,s, r1, o,r
                     optional match (o2:ObjectConcept)-[r2:contribute]->(a:ActionRepr) 
                     optional match (s)-[r3:influences]->(aff:Affordance)
                     match (a)-[r4:produces]->(aff) 
                     return e,s,r,r1,o,r2,a,r3,aff,r4 ,o2 
                """
            )
            nodes = list(result.graph()._nodes.values())
            rels = list(result.graph()._relationships.values())

            return nodes, rels

    def get_episode_graph_minigrid(self, episode_id):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}

                     match (e:Episode)-[r:has_state]->(s:StateT)-[r1:has_object]->(o:ObjectConcept)
                     where elementId(e)="{episode_id}"
                     with e,s, r1, o,r
                    match (s)-[r3:influences]->(aff:Affordance)
                     return e,s,r,r1,o,r3,aff
                """
            )
            nodes = list(result.graph()._nodes.values())
            rels = list(result.graph()._relationships.values())

            return nodes, rels

    def get_reduce_state_graph(self, state_id):
        query = f"""
        USE {self.memory_type}
        match (n:StateT)-[r1:has_object]->(o:ObjectConcept)
        where elementId(n)="{state_id}"
        with n, r1, o
        optional match (o1:ObjectConcept)-[r2:contribute]->(a:ActionRepr) 
        optional match (n)-[r3:influences]->(aff:Affordance)
        match (a)-[r4:produces]->(aff) 
        return n,r1,o,r2,a,r3,aff,r4 ,o1     
        """
        with self.driver.session() as session:
            result = session.run(query)
            nodes = list(result.graph()._nodes.values())
            rels = list(result.graph()._relationships.values())

            return nodes, rels

    def check_similar_states_time_based(self, t1, t2, sim_disim='sim',s_id=None):
        if sim_disim == 'sim':
            sim_string = 'sim > 0.6 and sim <1.0'
        else:
            sim_string = 'sim > 0 and sim <0.51'

        if s_id is None:
            s_check = f" where r.t={t1} "
        else:
            s_check = f" where r.t={t1} and elementId(s)='{s_id}'"

        query = f"""
                USE {self.memory_type}
               match (e:Episode)-[r:has_state]->(s:StateT) {s_check} with s, s.state_enc as enc
               match (e1:Episode)-[r1:has_state]->(s1:StateT) where r1.t={t2} with s, enc, s1, s1.state_enc as enc1
               with gds.similarity.euclidean(
                       enc,
                       enc1
                       ) as sim, s,s1

               where {sim_string}
               return elementId(s), elementId(s1), sim order by sim desc 
           """
        with self.driver.session() as session:
            result = session.run(query)
            df = pd.DataFrame([r.values() for r in result], columns=result.keys())
            return df, len(df)

        return None, []

    def get_obj_action_repr(self, object_id):
        query = f"""
        match(o:ObjectConcept)-[:contribute]-(a:ActionRepr) where elementId(o) ="{object_id}"
        return a
        """
        with self.driver.session() as session:
            result = session.run(query)
            nodes = list(result.graph()._nodes.values())
            return nodes


    def get_state_ids(self, count=1000, time=0):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    match (e:Episode)-[r:has_state]->(s:StateT) where r.t={time} return elementId(s) limit {count}
                """
            )
            result_ids = pd.DataFrame([r.values() for r in result], columns=result.keys())
            return result_ids


    def get_episode_ids(self, task,succesful='true', count=400):
        with self.driver.session() as session:
            if task is None and succesful is None:
                result = session.run(
                    f"""
                                    USE {self.memory_type}
                                    match (e:Episode) where e.set_embedding is NULL return elementId(e) limit {count}
                                """
                )
            else:

                result = session.run(
                f"""
                    USE {self.memory_type}
                    match (e:Episode) where e.task = "{task}" and e.succesfull_outcome={succesful} return elementId(e) limit {count}
                """
                )
            result_ids = pd.DataFrame([r.values() for r in result], columns=result.keys())
            return result_ids

    def get_all_ids(self, type='ObjectConcept'):
        with self.driver.session() as session:
            result = session.run(
                f"""
                    USE {self.memory_type}
                    match (o:{type}) return elementId(o)
                """
            )
            result_ids = pd.DataFrame([r.values() for r in result], columns=result.keys())
            return result_ids



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

    def get_objects_for_state(self, state_id):
        query = f""" USE {self.memory_type}
                match(s:StateT)-[:has_object]->(o:ObjectConcept) where elementId(s)="{state_id}" return elementId(o)"""
        with self.driver.session() as session:
            result = session.run(query)
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def get_states_for_episode(self, ep_id):
        query = f""" USER {self.memory_type}
                 MATCH (n:Episode)-[r:has_state]->(s:StateT) where elementId(n)={ep_id} return elementId(s),r.t"""
        with self.driver.session() as session:
            result = session.run(query)
            return pd.DataFrame([r.values() for r in result], columns=result.keys())



if __name__ == "__main__":
    # cs_memory = ConceptSpaceGDS()
    concept_space = ConceptSpaceGDS(memory_type="afftest")
    obj_non_zero_rew = concept_space.get_objects_associated_with_reward()
    # cs_memory.fetch_data("""
    # CALL db.propertyKeys
    # """)