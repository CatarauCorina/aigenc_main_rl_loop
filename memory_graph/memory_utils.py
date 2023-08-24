import os
from graphdatascience import GraphDataScience
import uuid
import numpy as np
import pandas as pd
from memory_graph.gds_concept_space import ConceptSpaceGDS


class WorkingMemory:
    DATABASE_URL = os.environ["NEO4J_BOLT_URL"]
    NEO_USER = os.environ['NEO_USER']
    NEO_PASS = os.environ['NEO_PASS']

    def __init__(self, default_name='objectConcept', which_db="longtermmemory"):
        self.gds = GraphDataScience(self.DATABASE_URL, auth=(self.NEO_USER, self.NEO_PASS))
        print(self.gds.version())
        self.project_name = default_name
        self.gds.set_database(which_db)
        self.default_name = default_name
        self.concept_space = ConceptSpaceGDS(memory_type=which_db)
        return

    def gds_init_project_catalog_objects(self):
        project_name = self.create_query_graph(self.default_name,'ObjectConcept', ['value'])
        print(project_name)
        return project_name

    def gds_init_project_catalog_states(self):
        project_name = self.create_query_graph("all_states", 'StateT', ['state_enc', 'parent_id'])
        print(project_name)
        return project_name

    def create_query_graph(self, project_name, object_type, properties):
        if self.check_if_graph_name_exists(project_name):
            project_name = project_name + str(uuid.uuid4())
        self.gds.run_cypher(
            f"""
            CALL gds.graph.project(
            '{project_name}',
            {{
            {object_type}: {{
            properties: {properties}
            }}
            }},
            '*'
            );
            """
        )
        return project_name

    def create_filtered_subgraph(self, subgraph_name, from_graph, condition, condition_value):
        if self.check_if_graph_name_exists(subgraph_name):
            subgraph_name = subgraph_name + str(uuid.uuid4())
        self.gds.run_cypher(
            f"""
                CALL gds.beta.graph.project.subgraph(
                    {subgraph_name},
                    {from_graph},
                    '{condition}={condition_value}',
                    '*'
                )
            """
        )
        return subgraph_name

    def check_if_graph_name_exists(self, name):
        name_exists = self.gds.run_cypher(f"""
            RETURN gds.graph.exists('{name}') AS name_exists
        """)
        return name_exists['name_exists'][0]

    def create_state_graph(self):
        return

    def compute_effect(self):
        return

    def compute_silhouette_best_k(self, project_name):
        k_silhouette = self.gds.run_cypher(
            f"""
            WITH range(2,14) AS kcol
            UNWIND kcol AS k
                CALL gds.beta.kmeans.stream({project_name},
                {{
                    nodeProperty: 'value',
                    computeSilhouette: true,
                    k: k
                }}
            ) YIELD nodeId, communityId, silhouette
            WITH k, avg(silhouette) AS avgSilhouette
            RETURN k, avgSilhouette
            """
        )
        best_k = k_silhouette['k'][k_silhouette['avgSilhouette'].idxmax()]
        return best_k

    def compute_wm_clusters(self, project_name):
        best_k = cs_memory.compute_silhouette_best_k(project_name)
        k_clustering_result = self.gds.run_cypher(
            f"""
            CALL gds.beta.kmeans.stream({project_name}, {{
            nodeProperty: 'value',
            k: {best_k},
            randomSeed: 42
            }})
            YIELD nodeId, communityId,distanceFromCentroid
            RETURN elementId(gds.util.asNode(nodeId)) AS id, communityId, distanceFromCentroid
            ORDER BY communityId, id ASC, distanceFromCentroid
            """
        )
        return k_clustering_result

    def compute_save_wm_clusters(self):
        best_k = cs_memory.compute_silhouette_best_k()
        k_clustering_result = self.gds.run_cypher(
            f"""
              CALL gds.beta.kmeans.write( 
                '{self.project_name}',
                {{
                    nodeProperty: 'value',
                    writeProperty: 'clusterVal',
                    k: {best_k}
                }}
            ) YIELD nodePropertiesWritten
               """
        )
        return k_clustering_result

    def compute_clusters_centroid(self):
        centroids = self.gds.run_cypher(
            """
                MATCH (u:ObjectConcept) WITH u.km13 AS cluster, u, range(0, 1000) AS ii 
                UNWIND ii AS i
                WITH cluster, i, avg(u.value[i]) AS avgVal
                ORDER BY cluster, i
                WITH cluster, collect(avgVal) AS avgEmbeddings
                MERGE (cl:Centroid{dimension: 1000, cluster: cluster})
                SET cl.embedding = avgEmbeddings
                RETURN cl;
            """
        )
        return centroids

    def check_if_node_exists(self, embedding, similarity_th=0.5, similarity_method='euclidean'):
        similar_objects = self.gds.run_cypher(
            f"""
                match(no:ObjectConcept) with 
                    no, gds.similarity.{similarity_method}(
                        {embedding},
                        no.value
                    ) as sim
                where sim > {similarity_th}
                return elementId(no),no.parent_id_state, sim order by sim limit 2
            """
        )
        return similar_objects

    def add_to_memory(self, encoded_state, current_screen, episode_id, timestep, reward):
        try:
            project_name = self.gds_init_project_catalog_objects()
        except:
            obj = self.concept_space.add_data('ObjectConcept')
            random_value = list(np.random.uniform(low=0.1, high=1, size=(1000,)))
            self.concept_space.update_node_by_id(obj['elementId(n)'][0],  random_value)
            self.concept_space.set_property(obj['elementId(n)'][0], 'ObjectConcept', 'att', 0.01)

            project_name = self.gds_init_project_catalog_objects()
        objects, state_id = self.concept_space.add_state_with_objects(
            encoded_state,
            current_screen,
            episode_id,
            timestep,
            reward.item()
        )
        state_split = int(state_id.split(':')[2])
        print(state_split)
        for obj in objects.squeeze(0).squeeze(0):
            obj_list = obj.tolist()
            similar_objects = self.check_if_node_exists(obj_list)
            if not similar_objects.empty:
                state_ids_list = list(similar_objects['no.parent_id_state'][0]) + [state_split]
                self.concept_space.set_property(similar_objects['elementId(no)'][0], 'ObjectConcept', 'parent_id_state', f'apoc.coll.toSet({state_ids_list})')
                obj_id = similar_objects['elementId(no)'][0]
            else:
                obj = self.concept_space.add_data('ObjectConcept')
                self.concept_space.set_property(obj['elementId(n)'][0], 'ObjectConcept', 'parent_id_state', [state_split])
                self.concept_space.update_node_by_id(obj['elementId(n)'][0], obj_list)
                self.concept_space.set_property(obj['elementId(n)'][0], 'ObjectConcept', 'att', 0.01)
                obj_id = obj['elementId(n)'][0]
            self.concept_space.match_state_add_node(state_id, obj_id)
        self.concept_space.close()
        return

    def compute_attention(self, time, episode_id, omega=0.1, beta=0.1, default_alpha=0.1):
        ep_id = int(episode_id.split(':')[2])
        reinforcer_and_sum = self.concept_space.objects_attention_and_reinforcer(time, ep_id)
        reinforcer_time_t = reinforcer_and_sum['reinforcer'][0]
        sum_all_obj = reinforcer_and_sum['sum(o.att)'][0]
        reward_current_time = reinforcer_and_sum['s.reward'][0]

        df_all_obj_att = self.concept_space.get_obj_att_at_time_t(time, ep_id)
        df_all_obj_att['not_obj_i'] = sum_all_obj - df_all_obj_att['o.att']

        obj_values_prev_time = self.concept_space.get_obj_att_values_prev_time(time, ep_id)
        if not obj_values_prev_time.empty:
            obj_values_prev_time['last_value_obj_i'] = [x[len(x) -1] for x in list(obj_values_prev_time['collect(o.att)'])]
            obj_values_prev_time['sum_val_obj_not_i'] = obj_values_prev_time['last_value_obj_i'].sum() - obj_values_prev_time['last_value_obj_i']
            obj_values_prev_time['alpha_obj_i'] = -omega*(reward_current_time - obj_values_prev_time['last_value_obj_i']) - (reward_current_time- obj_values_prev_time['sum_val_obj_not_i'])

            obj_to_update = pd.merge(df_all_obj_att, obj_values_prev_time, on="id_o")
            obj_to_update['delta_obj_i'] = obj_to_update['alpha_obj_i']*beta*(1-obj_to_update['o.att'])*reinforcer_time_t
            obj_to_update['new_value_obj_i'] = obj_to_update['o.att'] + obj_to_update['delta_obj_i']
        else:
            obj_to_update = df_all_obj_att
            obj_to_update['delta_obj_i'] = default_alpha * beta * (
                        1 - obj_to_update['o.att']) * reinforcer_time_t
            obj_to_update['new_value_obj_i'] = obj_to_update['o.att'] + obj_to_update['delta_obj_i']

        dict_values_to_update = list(obj_to_update[['id_o', 'new_value_obj_i']].to_dict('index').values())
        self.concept_space.update_objects_attention(dict_values_to_update)
        return





if __name__ == "__main__":
    cs_memory = WorkingMemory()
    clusters = cs_memory.compute_attention(2,"4:f668e156-00ed-4517-b866-5f67756e1d04:1538")
    print(clusters)
