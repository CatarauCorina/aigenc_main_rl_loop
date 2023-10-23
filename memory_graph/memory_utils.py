import os
import torch
from graphdatascience import GraphDataScience
import uuid
import numpy as np
import pandas as pd
from memory_graph.gds_concept_space import ConceptSpaceGDS


class WorkingMemory:
    DATABASE_URL = os.environ["NEO4J_BOLT_URL"]
    NEO_USER = os.environ['NEO_USER']
    NEO_PASS = os.environ['NEO_PASS']

    def __init__(self, default_name='objectConcept', which_db="wmtest"):
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

    def gds_init_project_catalog_action_repr(self):
        project_name = self.create_query_graph(self.default_name, 'ActionRepr', ['val'])
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

    def compute_silhouette_best_k(self, project_name, node_property='value'):
        k_silhouette = self.gds.run_cypher(
            f"""
            WITH range(2,14) AS kcol
            UNWIND kcol AS k
                CALL gds.beta.kmeans.stream({project_name},
                {{
                    nodeProperty: "{node_property}",
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

    def compute_wm_clusters(self, project_name, node_property='value'):
        best_k = cs_memory.compute_silhouette_best_k(project_name)
        k_clustering_result = self.gds.run_cypher(
            f"""
            CALL gds.beta.kmeans.stream({project_name}, {{
            nodeProperty: '{node_property}',
            k: {best_k},
            randomSeed: 42
            }})
            YIELD nodeId, communityId,distanceFromCentroid
            RETURN elementId(gds.util.asNode(nodeId)) AS id, communityId, distanceFromCentroid
            ORDER BY communityId, id ASC, distanceFromCentroid
            """
        )
        return k_clustering_result

    def compute_action_clusters(self, project_name, node_property='val', use_best_k=True):
        best_k = 11
        if use_best_k:
            best_k = self.compute_silhouette_best_k(project_name, node_property)

        query_string = f"""
            CALL gds.beta.kmeans.stream({project_name}, {{
            nodeProperty: '{node_property}',
            k: {best_k},
            randomSeed: 42,
            numberOfRestarts:5
            }})
            YIELD nodeId, communityId,distanceFromCentroid
            RETURN gds.util.asNode(nodeId).obj_type AS otype, gds.util.asNode(nodeId).{node_property} as emb, communityId, distanceFromCentroid
            ORDER BY communityId, otype ASC, emb, distanceFromCentroid
            """
        print(query_string)

        k_clustering_result = self.gds.run_cypher(query_string)
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

    def check_if_node_exists(self, embedding, similarity_th=0.5, similarity_method='euclidean', node_type='ObjectConcept'):
        similar_objects = self.gds.run_cypher(
            f"""
                match(no:{node_type}) with 
                    no, gds.similarity.{similarity_method}(
                        {embedding},
                        no.value
                    ) as sim
                where sim > {similarity_th}
                return elementId(no),no.parent_id_state, sim order by sim desc limit 2
            """
        )
        return similar_objects

    def add_affordance(self, position, reward, time_applied):
        add = self.gds.run_cypher(
            f"""
                MERGE (aff:Affordance {{ position:{position},reward:{reward},t:{time_applied} }})
                RETURN elementId(aff)
            """
        )
        return add

    def add_to_memory(self, encoded_state, current_screen, episode_id, timestep):
        self.init_project_catalogs(init_action=False)
        state_id = self.concept_space.add_state_with_objects(
            encoded_state,
            episode_id,
            timestep,
        )
        state_split = int(state_id.split(':')[2])
        print(state_split)
        for obj in current_screen.squeeze(0).squeeze(0):
            if torch.count_nonzero(obj).item() != 0:
                obj_list = obj.tolist()
                self.check_object_exists_and_add(obj_list, state_id)
        self.concept_space.close()
        return state_id

    def init_project_catalogs(self, init_action=False):
        try:
            project_name_obj = self.gds_init_project_catalog_objects()
        except:
            obj = self.concept_space.add_data('ObjectConcept')
            random_value = list(np.random.uniform(low=0.1, high=1, size=(16,)))
            self.concept_space.update_node_by_id(obj['elementId(n)'][0], random_value)
            self.concept_space.set_property(obj['elementId(n)'][0], 'ObjectConcept', 'att', 0.01)
            self.concept_space.set_property(obj['elementId(n)'][0], 'ObjectConcept', 'alpha', 0.1)
            self.concept_space.set_property(obj['elementId(n)'][0], 'ObjectConcept', 'all_att_values', [])

            project_name = self.gds_init_project_catalog_objects()
        if init_action:
            try:
                project_name_action = self.gds_init_project_catalog_action_repr()
            except:
                obj = self.concept_space.add_data('ActionRepr')
                random_value = list(np.random.uniform(low=0.1, high=1, size=(16,)))
                self.concept_space.update_node_by_id(obj['elementId(n)'][0], random_value)
                self.concept_space.set_property(obj['elementId(n)'][0], 'ActionRepr', 'obj_type', "init", is_string=True)

                project_name = self.gds_init_project_catalog_objects()

    def add_objects_action_set(self, objects, action_repr, state_id):
        state_split = int(state_id.split(':')[2])
        print(state_split)
        for obj in objects.squeeze(0).squeeze(0):
            if torch.count_nonzero(obj) != 0:
                obj_list = obj.tolist()
                obj_id = self.check_object_exists_and_add(obj_list, state_split)
        self.concept_space.close()

    def check_object_exists_and_add(self, obj_list, state_split):
        state_id = int(state_split.split(':')[2])
        similar_objects = self.check_if_node_exists(obj_list, similarity_th=0.8)
        if not similar_objects.empty:
            state_ids_list = list(similar_objects['no.parent_id_state'][0]) + [state_id]
            self.concept_space.set_property(similar_objects['elementId(no)'][0], 'ObjectConcept', 'parent_id_state',
                                            f'apoc.coll.toSet({state_ids_list})')
            obj_id = similar_objects['elementId(no)'][0]
        else:
            obj = self.concept_space.add_data('ObjectConcept')
            self.concept_space.set_property(obj['elementId(n)'][0], 'ObjectConcept', 'parent_id_state',
                                            [state_id])
            self.concept_space.set_property(obj['elementId(n)'][0], 'ObjectConcept', 'all_att_values', [])
            self.concept_space.update_node_by_id(obj['elementId(n)'][0], obj_list)
            self.concept_space.set_property(obj['elementId(n)'][0], 'ObjectConcept', 'att', 0.01)
            self.concept_space.set_property(obj['elementId(n)'][0], 'ObjectConcept', 'alpha', 0.1)
            obj_id = obj['elementId(n)'][0]
        self.concept_space.match_state_add_node(state_split, obj_id)
        return obj_id

    def check_action_repr_exists_and_add(self, action_repr, state_id, otype):
        similar_action_repr = self.check_if_node_exists(action_repr, node_type='ActionRepr', similarity_th=0.6)
        if not similar_action_repr.empty:
            state_ids_list = list(similar_action_repr['no.parent_id_state'][0]) + [state_id]
            self.concept_space.set_property(similar_action_repr['elementId(no)'][0], 'ActionRepr', 'parent_id_state',
                                            f'apoc.coll.toSet({state_ids_list})')
            action_id = similar_action_repr['elementId(no)'][0]
        else:
            action_node = self.concept_space.add_data('ActionRepr')
            action_id = action_node['elementId(n)'][0]
            self.concept_space.set_property(action_id, 'ActionRepr', 'value', action_repr)
            self.concept_space.set_property(action_id, 'ActionRepr', 'obj_type', otype, is_string=True)
            self.concept_space.set_property(action_id, 'ActionRepr', 'parent_id_state', [state_id])
            self.concept_space.set_property(action_id, 'ActionRepr', 'all_att_values', [])

            self.concept_space.set_property(action_id, 'ActionRepr', 'att', 0.01)
            self.concept_space.set_property(action_id, 'ActionRepr', 'alpha', 0.1)

        return action_id

    def add_object_action_repr(self, dict_interactions, state_id):
        self.init_project_catalogs(init_action=True)
        action_ids_tool_ids = {}

        state_split = int(state_id.split(':')[2])
        print(state_split)
        action_ids = []
        for interaction in dict_interactions:
            objects = interaction['objects_in_interaction']
            action_repr = interaction['interaction']
            action_id = self.check_action_repr_exists_and_add(action_repr.squeeze(0).tolist(),
                                                              state_split, interaction['tool_label'])
            action_ids_tool_ids[interaction['tool_id']] = action_id

            for obj in objects.squeeze(0).squeeze(0):
                if torch.count_nonzero(obj).item() != 0:
                    obj_list = obj.tolist()
                    obj_id = self.check_object_exists_and_add(obj_list, state_id)
                    self.concept_space.match_obj_add_action(obj_id, action_id)
                    action_ids.append(action_id)

        self.concept_space.close()
        return action_ids_tool_ids

    def match_action_affordance(self, action_tool_ids, applied_item, position, reward, time):
        aff_id = None
        try:
            action_id = action_tool_ids[applied_item]
            aff = self.add_affordance(position, reward, time)
            aff_id = aff['elementId(aff)'][0]
            self.concept_space.match_action_add_aff(action_id, aff_id)
        except:
            print(action_tool_ids.keys())
            print(applied_item)
            aff_id = None
        return aff_id

    def compute_attention(self, time, episode_id, omega=0.1, beta=0.6, default_alpha=0.1):
        ep_id = int(episode_id.split(':')[2])
        reinforcer_and_sum = self.concept_space.objects_attention_and_reinforcer(time, ep_id)
        reinforcer_time_t = abs(reinforcer_and_sum['reinforcer'][0])
        sum_all_obj = reinforcer_and_sum['sum(o.att)'][0]
        reward_current_time = reinforcer_and_sum['s.reward'][0]

        df_all_obj_att = self.concept_space.get_obj_att_at_time_t(time, ep_id)
        df_all_obj_att['not_obj_i'] = sum_all_obj - df_all_obj_att['o.att']

        obj_values_prev_time = self.concept_space.get_obj_att_values_prev_time(time, ep_id)
        if not obj_values_prev_time.empty:
            obj_values_prev_time['last_value_obj_i'] = obj_values_prev_time['prev_att']

            obj_values_prev_time['sum_val_obj_not_i'] = obj_values_prev_time['last_value_obj_i'].sum() - obj_values_prev_time['last_value_obj_i']
            obj_values_prev_time['delta_alpha_obj_i'] = -omega*(reward_current_time - obj_values_prev_time['last_value_obj_i']) - (reward_current_time- obj_values_prev_time['sum_val_obj_not_i'])
            obj_values_prev_time['alpha_obj_i_temp'] = obj_values_prev_time['alpha'] + obj_values_prev_time['delta_alpha_obj_i']
            obj_values_prev_time['alpha_obj_i'] = obj_values_prev_time['alpha_obj_i_temp']
            obj_values_prev_time.loc[obj_values_prev_time["alpha_obj_i_temp"] <= 0.05, 'alpha_obj_i'] = 0.05
            obj_values_prev_time.loc[obj_values_prev_time["alpha_obj_i_temp"] >= 1, 'alpha_obj_i'] = 1

            obj_to_update = pd.merge(df_all_obj_att, obj_values_prev_time, on="id_o")
            obj_to_update['delta_obj_i'] = obj_to_update['alpha_obj_i']*beta*(1-obj_to_update['o.att'])*reinforcer_time_t
            obj_to_update['new_value_obj_i'] = obj_to_update['o.att'] + obj_to_update['delta_obj_i']
            existing_vals = obj_to_update['att_values'].to_list()
            new_vals = [[new_val] for new_val in obj_to_update['new_value_obj_i'].to_list()]
            combined_list = [existing_vals[idx]+new_vals[idx] for idx, x in enumerate(existing_vals)]
            obj_to_update['att_values'] = combined_list

        else:
            obj_to_update = df_all_obj_att
            obj_to_update['delta_obj_i'] = default_alpha * beta * (
                        1 - obj_to_update['o.att']) * reinforcer_time_t
            obj_to_update['new_value_obj_i'] = obj_to_update['o.att'] + obj_to_update['delta_obj_i']
            existing_vals = obj_to_update['att_values'].to_list()
            new_vals = [[new_val] for new_val in obj_to_update['new_value_obj_i'].to_list()]
            combined_list = [existing_vals[idx] + new_vals[idx] for idx, x in enumerate(existing_vals)]
            obj_to_update['att_values'] = combined_list
            obj_to_update['alpha_obj_i'] = obj_to_update['alpha']

        dict_values_to_update = list(obj_to_update[['id_o', 'new_value_obj_i', 'att_values','alpha_obj_i']].to_dict('index').values())
        self.concept_space.update_objects_attention(dict_values_to_update)
        return





if __name__ == "__main__":
    cs_memory = WorkingMemory()
    #clusters = cs_memory.compute_attention(2,"4:f668e156-00ed-4517-b866-5f67756e1d04:1538")
    res = cs_memory.concept_space.get_attention_for_episode()
    print(res)
