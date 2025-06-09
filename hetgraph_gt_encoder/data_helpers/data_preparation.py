import os
import re
import torch
import sys
import pickle
import random
import numpy as np
from random import choice
import scipy.sparse as sp

print(os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))

import memory_graph
from memory_graph.gds_concept_space import ConceptSpaceGDS


class StateLoader:

    def __init__(self, nr_mps, mps, use_memory='outcomesmall', tasks=None):
        self.nr_mps = nr_mps
        self.cs_memory = ConceptSpaceGDS(memory_type=use_memory)
        self.embedding_mapping = {
            'ObjectConcept':'value',
            'ActionRepr':'value',
            'Affordance': 'outcome',
            'StateT': 'state_enc'
        }
        if tasks is None:
            self.tasks = ["CreateLevelPush-v0",
                       "CreateLevelBuckets-v0",
                       "CreateLevelBasket-v0","CreateLevelBelt-v0",
                       "CreateLevelObstacle-v0"]
        else:
            self.tasks = tasks

    def process_state_node_data(self, node):
        node_type = set(node.labels).pop()
        node_id = node.element_id
        try:
            node_embedding = node._properties[self.embedding_mapping[node_type]]
        except:
            node_embedding = None
        return node_type, node_id, node_embedding

    def get_state_data(self, time=0):
        all_ids = self.cs_memory.get_state_ids(time=time)
        for index, state_id in all_ids.iterrows():
            st_data = {
                'StateT': [],
                'ObjectConcept': [],
                'Affordance': [],
                'ActionRepr': [],
                'ObjectConceptRel': [],
                'ActionReprRel': [],
                'AffordanceRel': [],
                'full_rels': None
            }

            # st_data_reduced = {
            #     'StateT': [],
            #     'ObjectConcept': [],
            #     'Affordance': [],
            #     'ActionRepr': [],
            #     'ObjectConceptRel': [],
            #     'ActionReprRel': [],
            #     'AffordanceRel': [],
            #     'full_rels': None
            # }

            st_nodes, st_rel = self.cs_memory.get_reduce_state_graph(state_id['elementId(s)'])
            # reduces_st_nodes, reduced_st_rel = self.cs_memory.get_reduce_state_graph(state_id['elementId(s)'])
            for node in st_nodes:
                node_type, node_id, node_emb = self.process_state_node_data(node)
                st_data[node_type].append((node_id, node_emb))

            # for node in reduces_st_nodes:
            #     node_type, node_id, node_emb = self.process_state_node_data(node)
            #     st_data_reduced[node_type].append((node_id, node_emb))

            object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in st_rel if set(rel.end_node.labels).pop() == 'ObjectConcept']
            action_repr_rel =[(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in st_rel if set(rel.end_node.labels).pop() == 'ActionRepr']
            # aff_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in st_rel if set(rel.end_node.labels).pop() == 'Affordance']

            # object_rels_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in reduced_st_rel if
            #                set(rel.end_node.labels).pop() == 'ObjectConcept']
            # action_repr_rel_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in reduced_st_rel if
            #                    set(rel.end_node.labels).pop() == 'ActionRepr']
            aff_rels_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in st_rel if
                        set(rel.end_node.labels).pop() == 'Affordance']

            st_data['ObjectConceptRel'] = object_rels
            st_data['ActionReprRel'] = action_repr_rel
            st_data['AffordanceRel'] = aff_rels_red
            #st_data['full_rels'] = st_rel

            # st_data_reduced['ObjectConceptRel'] = object_rels_red
            # st_data_reduced['ActionReprRel'] = action_repr_rel_red
            # st_data_reduced['AffordanceRel'] = aff_rels_red
            #st_data_reduced['full_rels'] = reduced_st_rel
            id = str(state_id['elementId(s)']).replace(':','-')
            directory_path = f'{os.path.join(os.path.dirname(os.getcwd()),"states_data_time_2")}\\{time}\\'
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            with open(f'{directory_path}{id}.pickle', 'wb') as handle:
                pickle.dump(st_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(f'{os.path.join(os.path.dirname(os.getcwd()),"states_data")}\\states_data_reduced_t{time}_{index}.pickle', 'wb') as handle:
            #     pickle.dump(st_data_reduced, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return


    def get_episode_data(self, task='CreateLevelPush-v0', succesfull='true'):
        all_ids = self.cs_memory.get_episode_ids(task=task,succesful=succesfull)
        for index, ep_id in all_ids.iterrows():
            ep_data = {
                'Episode': [],
                'StateT': [],
                'ObjectConcept': [],
                'Affordance': [],
                'ActionRepr': [],
                "StateTRel": [],
                'ObjectConceptRel': [],
                'ActionReprRel': [],
                'AffordanceRel': [],
                'full_rels': None
            }


            ep_nodes, ep_rel = self.cs_memory.get_episode_graph(ep_id['elementId(e)'])
            # reduces_st_nodes, reduced_st_rel = self.cs_memory.get_reduce_state_graph(state_id['elementId(s)'])
            for node in ep_nodes:
                node_type, node_id, node_emb = self.process_state_node_data(node)
                ep_data[node_type].append((node_id, node_emb))


            state_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if set(rel.end_node.labels).pop() == 'StateT']
            object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if set(rel.end_node.labels).pop() == 'ObjectConcept']
            action_repr_rel =[(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if set(rel.end_node.labels).pop() == 'ActionRepr']

            aff_rels_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                        set(rel.end_node.labels).pop() == 'Affordance']

            ep_data['ObjectConceptRel'] = object_rels
            ep_data['ActionReprRel'] = action_repr_rel
            ep_data['AffordanceRel'] = aff_rels_red
            ep_data['StateTRel'] = state_rels

            id = str(ep_id['elementId(e)']).replace(':','-')
            directory_path = f'{os.path.join(os.path.dirname(os.getcwd()),"ep_data_incomplete", task, succesfull)}\\'
            if len(ep_data['StateT']) > 0:
                # print(len(ep_data['StateT']))
                # ep_data['StateT'].remove(random.choice(ep_data['StateT']))
                # ep_data['StateT'].remove(random.choice(ep_data['StateT']))

                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                with open(f'{directory_path}{id}.pickle', 'wb') as handle:
                    pickle.dump(ep_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return



    def get_episode_data_minigrid(self, task='MiniGrid-Empty-5x5-v0', succesfull='true'):
        all_ids = self.cs_memory.get_episode_ids(task=task,succesful=succesfull)
        for index, ep_id in all_ids.iterrows():
            ep_data = {
                'Episode': [],
                'StateT': [],
                'ObjectConcept': [],
                'Affordance': [],
                "StateTRel": [],
                'ObjectConceptRel': [],
                'AffordanceRel': [],
                'full_rels': None
            }


            ep_nodes, ep_rel = self.cs_memory.get_episode_graph_minigrid(ep_id['elementId(e)'])
            # reduces_st_nodes, reduced_st_rel = self.cs_memory.get_reduce_state_graph(state_id['elementId(s)'])
            for node in ep_nodes:
                node_type, node_id, node_emb = self.process_state_node_data(node)
                ep_data[node_type].append((node_id, node_emb))


            state_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if set(rel.end_node.labels).pop() == 'StateT']
            object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if set(rel.end_node.labels).pop() == 'ObjectConcept']

            aff_rels_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                        set(rel.end_node.labels).pop() == 'Affordance']

            ep_data['ObjectConceptRel'] = object_rels
            ep_data['AffordanceRel'] = aff_rels_red
            ep_data['StateTRel'] = state_rels

            id = str(ep_id['elementId(e)']).replace(':','-')
            directory_path = f'{os.path.join(os.path.dirname(os.getcwd()),"ep_data_minigrid", task, succesfull)}\\'
            if len(ep_data['StateT']) > 0:
                # print(len(ep_data['StateT']))
                # ep_data['StateT'].remove(random.choice(ep_data['StateT']))
                # ep_data['StateT'].remove(random.choice(ep_data['StateT']))

                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                with open(f'{directory_path}{id}.pickle', 'wb') as handle:
                    pickle.dump(ep_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def remove_rand_data(self, task='CreateLevelPush-v0', succesfull='true'):
        all_ids = self.cs_memory.get_episode_ids(task=task, succesful=succesfull)
        for index, ep_id in all_ids.iterrows():
            ep_data = {
                'Episode': [],
                'StateT': [],
                'ObjectConcept': [],
                'Affordance': [],
                'ActionRepr': [],
                "StateTRel": [],
                'ObjectConceptRel': [],
                'ActionReprRel': [],
                'AffordanceRel': [],
                'full_rels': None
            }

            # Retrieve nodes and relations
            ep_nodes, ep_rel = self.cs_memory.get_episode_graph(ep_id['elementId(e)'])
            for node in ep_nodes:
                node_type, node_id, node_emb = self.process_state_node_data(node)
                ep_data[node_type].append((node_id, node_emb))

            # Separate relations by type
            state_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                          set(rel.end_node.labels).pop() == 'StateT']
            object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                           set(rel.end_node.labels).pop() == 'ObjectConcept']
            action_repr_rel = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                               set(rel.end_node.labels).pop() == 'ActionRepr']
            aff_rels_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                            set(rel.end_node.labels).pop() == 'Affordance']

            # Store relations
            ep_data['ObjectConceptRel'] = object_rels
            ep_data['ActionReprRel'] = action_repr_rel
            ep_data['AffordanceRel'] = aff_rels_red
            ep_data['StateTRel'] = state_rels

            # Remove 40% of the states and their relations
            id = str(ep_id['elementId(e)']).replace(':', '-')
            directory_path = f'{os.path.join(os.path.dirname(os.getcwd()), "ep_data_incomplete_40", task, succesfull)}\\'

            if len(ep_data['StateT']) > 0:
                # Calculate the number of states to remove (40% of total states)
                num_to_remove = max(1, int(40 * len(ep_data['StateT']) / 100))
                print(num_to_remove)
                states_to_remove = random.sample(ep_data['StateT'], num_to_remove)
                state_ids_to_remove = [state[0] for state in states_to_remove]

                for state in states_to_remove:
                    ep_data['StateT'].remove(state)

                # Remove relations connected to the removed states
                ep_data['StateTRel'] = [rel for rel in ep_data['StateTRel'] if
                                        rel[0] not in state_ids_to_remove and rel[1] not in state_ids_to_remove]
                ep_data['ObjectConceptRel'] = [rel for rel in ep_data['ObjectConceptRel'] if
                                               rel[0] not in state_ids_to_remove and rel[1] not in state_ids_to_remove]
                ep_data['ActionReprRel'] = [rel for rel in ep_data['ActionReprRel'] if
                                            rel[0] not in state_ids_to_remove and rel[1] not in state_ids_to_remove]
                ep_data['AffordanceRel'] = [rel for rel in ep_data['AffordanceRel'] if
                                            rel[0] not in state_ids_to_remove and rel[1] not in state_ids_to_remove]

                # Remove affordances connected to the removed states
                ep_data['Affordance'] = [aff for aff in ep_data['Affordance'] if aff[0] not in state_ids_to_remove]

                # Save the modified data
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                with open(f'{directory_path}{id}.pickle', 'wb') as handle:
                    pickle.dump(ep_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def reorder_states(self, task='CreateLevelPush-v0', succesfull='true'):
        all_ids = self.cs_memory.get_episode_ids(task=task, succesful=succesfull)
        for index, ep_id in all_ids.iterrows():
            ep_data = {
                'Episode': [],
                'StateT': [],
                'ObjectConcept': [],
                'Affordance': [],
                'ActionRepr': [],
                "StateTRel": [],
                'ObjectConceptRel': [],
                'ActionReprRel': [],
                'AffordanceRel': [],
                'full_rels': None
            }

            # Retrieve nodes and relations
            ep_nodes, ep_rel = self.cs_memory.get_episode_graph(ep_id['elementId(e)'])
            for node in ep_nodes:
                node_type, node_id, node_emb = self.process_state_node_data(node)
                ep_data[node_type].append((node_id, node_emb))

            # Separate relations by type
            state_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                          set(rel.end_node.labels).pop() == 'StateT']
            object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                           set(rel.end_node.labels).pop() == 'ObjectConcept']
            action_repr_rel = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                               set(rel.end_node.labels).pop() == 'ActionRepr']
            aff_rels_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                            set(rel.end_node.labels).pop() == 'Affordance']

            # Store relations
            ep_data['ObjectConceptRel'] = object_rels
            ep_data['ActionReprRel'] = action_repr_rel
            ep_data['AffordanceRel'] = aff_rels_red
            ep_data['StateTRel'] = state_rels

            # Reorder states randomly
            if len(ep_data['StateT']) > 0:
                original_states = ep_data['StateT']
                random.shuffle(original_states)
                ep_data['StateT'] = original_states

                # Update relations to match reordered states
                state_ids = {state[0]: idx for idx, state in enumerate(ep_data['StateT'])}
                ep_data['StateTRel'] = [
                    (state_ids[rel[0]], state_ids[rel[1]]) for rel in ep_data['StateTRel']
                    if rel[0] in state_ids and rel[1] in state_ids
                ]

                # Save the modified data
                id = str(ep_id['elementId(e)']).replace(':', '-')
                directory_path = f'{os.path.join(os.path.dirname(os.getcwd()), "ep_data_reordered", task, succesfull)}\\'

                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                with open(f'{directory_path}{id}.pickle', 'wb') as handle:
                    pickle.dump(ep_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return



    def flatten_graph_data(self, task='CreateLevelPush-v0', succesfull='true'):
        all_ids = self.cs_memory.get_episode_ids(task=task, succesful=succesfull)
        for index, ep_id in all_ids.iterrows():
            ep_data = {
                'Episode': [],
                'StateT': [],
                'ObjectConcept': [],
                'Affordance': [],
                'ActionRepr': [],
                "StateTRel": [],
                'ObjectConceptRel': [],
                'ActionReprRel': [],
                'AffordanceRel': [],
                'full_rels': None
            }

            # Retrieve nodes and relations
            ep_nodes, ep_rel = self.cs_memory.get_episode_graph(ep_id['elementId(e)'])
            for node in ep_nodes:
                node_type, node_id, node_emb = self.process_state_node_data(node)
                ep_data[node_type].append((node_id, node_emb))

            # Remove all relations by setting them to empty lists
            ep_data['StateTRel'] = []
            ep_data['ObjectConceptRel'] = []
            ep_data['ActionReprRel'] = []
            ep_data['AffordanceRel'] = []

            # Save the flattened data
            id = str(ep_id['elementId(e)']).replace(':', '-')
            directory_path = f'{os.path.join(os.path.dirname(os.getcwd()), "ep_data_flattened", task, succesfull)}\\'

            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            with open(f'{directory_path}{id}.pickle', 'wb') as handle:
                pickle.dump(ep_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def remove_low_level_nodes(self, task='CreateLevelPush-v0', succesfull='true'):
        all_ids = self.cs_memory.get_episode_ids(task=task, succesful=succesfull)
        for index, ep_id in all_ids.iterrows():
            ep_data = {
                'Episode': [],
                'StateT': [],
                'ObjectConcept': [],
                'Affordance': [],
                'ActionRepr': [],
                "StateTRel": [],
                'ObjectConceptRel': [],
                'ActionReprRel': [],
                'AffordanceRel': [],
                'full_rels': None
            }

            # Retrieve nodes and relations
            ep_nodes, ep_rel = self.cs_memory.get_episode_graph(ep_id['elementId(e)'])
            for node in ep_nodes:
                node_type, node_id, node_emb = self.process_state_node_data(node)
                ep_data[node_type].append((node_id, node_emb))

            # Separate relations by type
            state_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                          set(rel.end_node.labels).pop() == 'StateT']
            object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                           set(rel.end_node.labels).pop() == 'ObjectConcept']
            action_repr_rel = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                               set(rel.end_node.labels).pop() == 'ActionRepr']
            aff_rels_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                            set(rel.end_node.labels).pop() == 'Affordance']

            # Store relations
            ep_data['ObjectConceptRel'] = object_rels
            ep_data['ActionReprRel'] = action_repr_rel
            ep_data['AffordanceRel'] = aff_rels_red
            ep_data['StateTRel'] = state_rels

            # Remove all ObjectConcept and ActionRepr nodes and their relations
            id = str(ep_id['elementId(e)']).replace(':', '-')
            directory_path = f'{os.path.join(os.path.dirname(os.getcwd()), "ep_data_no_low_level", task, succesfull)}\\'

            # Identify ObjectConcept and ActionRepr nodes to remove
            low_level_nodes = ep_data['ObjectConcept'] + ep_data['ActionRepr']
            low_level_ids = [node[0] for node in low_level_nodes]

            # Remove the nodes
            ep_data['ObjectConcept'] = []
            ep_data['ActionRepr'] = []

            # Remove relations connected to the low-level nodes
            ep_data['ObjectConceptRel'] = [rel for rel in ep_data['ObjectConceptRel'] if
                                           rel[0] not in low_level_ids and rel[1] not in low_level_ids]
            ep_data['ActionReprRel'] = [rel for rel in ep_data['ActionReprRel'] if
                                        rel[0] not in low_level_ids and rel[1] not in low_level_ids]
            ep_data['AffordanceRel'] = [rel for rel in ep_data['AffordanceRel'] if
                                        rel[0] not in low_level_ids and rel[1] not in low_level_ids]

            # Save the modified data
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            with open(f'{directory_path}{id}.pickle', 'wb') as handle:
                pickle.dump(ep_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def get_state_data_by_id(self, state_id):
        st_data = {
            'StateT': [],
            'ObjectConcept': [],
            'Affordance': [],
            'ActionRepr': [],
            'ObjectConceptRel': [],
            'ActionReprRel': [],
            'AffordanceRel': [],
            'full_rels': None
        }

        st_nodes, st_rel = self.cs_memory.get_state_graph(state_id)
        reduces_st_nodes, reduced_st_rel = self.cs_memory.get_reduce_state_graph(state_id)
        for node in st_nodes:
            node_type, node_id, node_emb = self.process_state_node_data(node)
            st_data[node_type].append((node_id, node_emb))

        object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in st_rel if
                       set(rel.end_node.labels).pop() == 'ObjectConcept']
        action_repr_rel = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in st_rel if
                           set(rel.end_node.labels).pop() == 'ActionRepr']
        aff_rels_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in reduced_st_rel if
                        set(rel.end_node.labels).pop() == 'Affordance']

        st_data['ObjectConceptRel'] = object_rels
        st_data['ActionReprRel'] = action_repr_rel
        st_data['AffordanceRel'] = aff_rels_red

        obj_ids = self.cs_memory.get_ids_node('ObjectConcept')['elementId(o)'].tolist()
        action_ids = self.cs_memory.get_ids_node('ActionRepr')['elementId(o)'].tolist()
        affordance_ids = self.cs_memory.get_ids_node('Affordance')['elementId(o)'].tolist()

        states_data, full_state = self.fetch_data_for_states_pickle("", obj_ids, affordance_ids, data=st_data)
        feats = states_data[0][0]
        nei_index = states_data[0][1]
        mps = self.generate_mps_st(nei_index, affordance_ids, obj_ids, action_ids, full_state)
        return feats, nei_index, mps

    def load_state_data(self, file_name, full_path=None):
        file_path = 'states_data.pickle'
        if full_path:
            with open(os.path.abspath(full_path), 'rb') as handle:
                loaded_data = pickle.load(handle)
            return loaded_data

        with open(f'{os.path.join(os.getcwd(), "states_data_2")}\\{file_name}','rb') as handle:
            loaded_data = pickle.load(handle)
        return loaded_data

    def load_ep_data(self, file_name, full_path=None):
        print(file_name)
        print(full_path)
        if full_path:
            try:
                with open(os.path.abspath(full_path), 'rb') as handle:
                    loaded_data = pickle.load(handle)
                return loaded_data
            except:
                print(full_path)
                with open(full_path, 'rb') as handle:
                    loaded_data = pickle.load(handle)
                return loaded_data


        with open(f'{os.path.join(os.getcwd(), "ep_data")}\\{file_name}', 'rb') as handle:
            loaded_data = pickle.load(handle)
        return loaded_data

    def load_train_data(self):
        file_path_feats = 'feats'
        file_path_nei_index = 'nei_index'
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "train_data_states")}\\{file_path_feats}.pickle',
                  'rb') as handle:
            data_features = pickle.load(handle)

        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "train_data_states")}\\{file_path_nei_index}.pickle',
                  'rb') as handle:
            nei_index = pickle.load(handle)
        return data_features, nei_index

    def get_feats_state_st(self, file_name, data=None, full_path=None):
        feats = []
        if data is None:
            data = self.load_state_data(file_name, full_path=full_path)
        states_dict = dict(data['StateT'])
        feats.append(torch.tensor(list(states_dict.values())).cuda())
        object_dict = dict(data['ObjectConcept'])
        feats.append(torch.tensor(list(object_dict.values())).cuda())
        affordance = dict(data['Affordance'])
        if len(list(affordance.values())) == 1 and list(affordance.values())[0] is None:
            feats.append(torch.tensor([]).cuda())
        else:
            feats.append(torch.tensor(list(affordance.values())).cuda())
        action_repr = dict(data['ActionRepr'])
        feats.append(torch.tensor(list(action_repr.values())).cuda())
        return feats, data

    def get_feats_episode(self, file_name, data=None, full_path=None):
        feats = []
        if data is None:
            data = self.load_ep_data(file_name, full_path=full_path)
        # ep_dict = dict(data['Episode'])
        # feats.append(torch.tensor(list(ep_dict.values())).cuda())
        states_dict = dict(data['StateT'])
        feats.append(torch.tensor(list(states_dict.values())).cuda())
        object_dict = dict(data['ObjectConcept'])
        feats.append(torch.tensor(list(object_dict.values())).cuda())
        affordance = dict(data['Affordance'])
        if len(list(affordance.values())) == 1 and list(affordance.values())[0] is None:
            feats.append(torch.tensor([]).cuda())
        else:
            cleaned_list = [item for item in list(affordance.values()) if item is not None]

            feats.append(torch.tensor(cleaned_list).cuda())


            # feats = [
            #     torch.rand(513).cuda() if value is None else torch.tensor(value).cuda()
            #     for value in affordance.values()
            # ]
        if 'ActionRepr' in data.keys():
            action_repr = dict(data['ActionRepr'])
            feats.append(torch.tensor(list(action_repr.values())).cuda())
        return feats, data

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def generate_mps_st(self, nei_objs, all_aff_keys, all_obj_keys, action_keys, full_state_data):
        if len(nei_objs) > 0:
            count_objs = nei_objs[0][0].size()
            size_of_objects = len(all_obj_keys)
            size_action_repr = len(action_keys)
            size_affordances = len(all_aff_keys)
            obj_action_matrix = (size_of_objects, size_action_repr)
            action_aff_matrix = (size_action_repr, size_affordances)
            action_objs_x = []
            action_objs_y = []
            for idx1, idx2 in full_state_data['ActionReprRel']:
                if idx1 in all_obj_keys:
                    obj_key = idx1
                    action_key = idx2
                if idx2 in all_obj_keys:
                    obj_key = idx2
                    action_key = idx1
                position_x = all_obj_keys.index(obj_key)
                position_y = action_keys.index(action_key)
                action_objs_x.append(position_x)
                action_objs_y.append(position_y)

            action_aff_x = []
            action_aff_y = []
            for idx1, idx2 in full_state_data['AffordanceRel']:
                if idx1 in action_keys:
                    action_key = idx1
                    aff_key = idx2
                    position_x = action_keys.index(action_key)
                    position_y = all_aff_keys.index(aff_key)
                    action_aff_x.append(position_x)
                    action_aff_y.append(position_y)
                elif idx2 in action_keys:
                    action_key = idx2
                    aff_key = idx1
                    position_x = action_keys.index(action_key)
                    position_y = all_aff_keys.index(aff_key)
                    action_aff_x.append(position_x)
                    action_aff_y.append(position_y)


            st_o = sp.coo_matrix((np.ones(count_objs[0]), (np.zeros(count_objs[0]), nei_objs[0][0].cpu().numpy())), shape=(1, size_of_objects)).toarray()
            obj_action = sp.coo_matrix((np.ones(len(full_state_data['ActionReprRel'])), (action_objs_x, action_objs_y)), shape=obj_action_matrix).toarray()
            action_aff = sp.coo_matrix((np.ones(len(action_aff_x)), (action_aff_x, action_aff_y)), shape=action_aff_matrix).toarray()

            st_obj_action = np.matmul(st_o, obj_action) > 0
            st_obj_action_mp = sp.coo_matrix(st_obj_action)

            st_obj_action_aff = np.matmul(st_obj_action, action_aff) > 0
            st_obj_action_aff_mp = sp.coo_matrix(st_obj_action_aff)

            stobjaction_mp = self.sparse_mx_to_torch_sparse_tensor(st_obj_action_mp)
            stobjactionaff_mp = self.sparse_mx_to_torch_sparse_tensor(st_obj_action_aff_mp)
            return [stobjaction_mp, stobjactionaff_mp]
        else:
            return []


    def get_all_keys(self, has_action_repr=True):
        st_ids = self.cs_memory.get_all_ids('StateT')
        obj_ids = self.cs_memory.get_all_ids('ObjectConcept')
        aff_ids = self.cs_memory.get_all_ids('Affordance')
        if has_action_repr:
            act_ids = self.cs_memory.get_all_ids('ActionRepr')
            return list(st_ids['elementId(o)']), list(obj_ids['elementId(o)']), list(act_ids['elementId(o)']), list(aff_ids['elementId(o)'])
        return list(st_ids['elementId(o)']), list(obj_ids['elementId(o)']), list(
            aff_ids['elementId(o)'])

    def generate_mps_episode(self, nei_states, full_state_data, has_action_repr=True):
        if len(nei_states) > 0:
            if has_action_repr:
                all_state_keys, all_obj_keys, action_keys, all_aff_keys = self.get_all_keys(has_action_repr=has_action_repr)
                size_of_objects = len(all_obj_keys)
                size_affordances = len(all_aff_keys)

                size_action_repr = len(action_keys)
                obj_action_matrix = (size_of_objects, size_action_repr)
                action_aff_matrix = (size_action_repr, size_affordances)




            else:
                all_state_keys, all_obj_keys, all_aff_keys = self.get_all_keys(has_action_repr=has_action_repr)
                size_of_objects = len(all_obj_keys)
                size_affordances = len(all_aff_keys)

            count_states = nei_states[0][0].size()

            size_states = len(all_state_keys)
            state_obj_matrix = (size_states, size_of_objects)
            state_aff_matrix = (size_states, size_affordances)
            aff_state_matrix = (size_affordances, size_states)


            state_objs_x = []
            state_objs_y = []
            nr_state_obj_rels = 0
            for idx1, idx2 in full_state_data['ObjectConceptRel']:
                if idx1 in all_state_keys and idx2 in all_obj_keys:
                    state_key = idx1
                    object_key = idx2
                    position_x = all_state_keys.index(state_key)
                    position_y = all_obj_keys.index(object_key)
                    state_objs_x.append(position_x)
                    state_objs_y.append(position_y)
                    nr_state_obj_rels +=1
                elif idx2 in all_state_keys and idx1 in all_obj_keys:
                    state_key = idx2
                    object_key = idx1
                    position_x = all_state_keys.index(state_key)
                    position_y = all_obj_keys.index(object_key)
                    state_objs_x.append(position_x)
                    state_objs_y.append(position_y)
                    nr_state_obj_rels +=1



            action_objs_x = []
            action_objs_y = []
            nr_action_obj_rels = 0


            if has_action_repr:
                for idx1, idx2 in full_state_data['ActionReprRel']:
                    if idx1 in all_obj_keys and idx2 in action_keys:
                        obj_key = idx1
                        action_key = idx2
                        position_x = all_obj_keys.index(obj_key)
                        position_y = action_keys.index(action_key)
                        action_objs_x.append(position_x)
                        action_objs_y.append(position_y)
                        nr_action_obj_rels +=1

                    elif idx2 in all_obj_keys and idx1 in action_keys:
                        obj_key = idx2
                        action_key = idx1
                        position_x = all_obj_keys.index(obj_key)
                        position_y = action_keys.index(action_key)
                        action_objs_x.append(position_x)
                        action_objs_y.append(position_y)
                        nr_action_obj_rels +=1



            action_aff_x = []
            action_aff_y = []
            nr_action_aff_rels = 0

            state_aff_x = []
            state_aff_y = []
            nr_state_aff_rels = 0

            for idx1, idx2 in full_state_data['AffordanceRel']:
                if has_action_repr and idx1 in action_keys and idx2 in all_aff_keys:
                    action_key = idx1
                    aff_key = idx2
                    position_x = action_keys.index(action_key)
                    position_y = all_aff_keys.index(aff_key)
                    action_aff_x.append(position_x)
                    action_aff_y.append(position_y)
                    nr_action_aff_rels +=1

                elif has_action_repr and idx2 in action_keys and idx1 in all_aff_keys:
                    action_key = idx2
                    aff_key = idx1
                    position_x = action_keys.index(action_key)
                    position_y = all_aff_keys.index(aff_key)
                    action_aff_x.append(position_x)
                    action_aff_y.append(position_y)
                    nr_action_aff_rels += 1
                elif idx1 in all_state_keys and idx2 in all_aff_keys:
                    aff_key = idx2
                    state_key = idx1

                    position_x = all_state_keys.index(state_key)
                    position_y = all_aff_keys.index(aff_key)
                    state_aff_x.append(position_x)
                    state_aff_y.append(position_y)
                    nr_state_aff_rels += 1
                elif idx2 in all_state_keys and idx1 in all_aff_keys:
                    aff_key = idx1
                    state_key = idx2

                    position_x = all_state_keys.index(state_key)
                    position_y = all_aff_keys.index(aff_key)
                    state_aff_x.append(position_x)
                    state_aff_y.append(position_y)
                    nr_state_aff_rels +=1



            ep_state = sp.coo_matrix((np.ones(count_states[0]), (np.zeros(count_states[0]), nei_states[0][0].cpu().numpy())), shape=(1, size_states)).toarray()
            st_obj = sp.coo_matrix((np.ones(nr_state_obj_rels), (state_objs_x, state_objs_y)), shape=state_obj_matrix).toarray()
            st_aff = sp.coo_matrix((np.ones(nr_state_aff_rels), (state_aff_x, state_aff_y)), shape=state_aff_matrix).toarray()
            aff_st = sp.coo_matrix((np.ones(nr_state_aff_rels), (state_aff_y, state_aff_x)), shape=aff_state_matrix).toarray()

            if has_action_repr:
                obj_action = sp.coo_matrix((np.ones(nr_action_obj_rels), (action_objs_x, action_objs_y)), shape=obj_action_matrix).toarray()
                action_aff = sp.coo_matrix((np.ones(nr_action_aff_rels), (action_aff_x, action_aff_y)), shape=action_aff_matrix).toarray()

            ep_st_aff = np.matmul(ep_state, st_aff) > 0
            ep_st_aff_mp = sp.coo_matrix(ep_st_aff)

            ep_st_obj = np.matmul(ep_state, st_obj) > 0
            ep_st_obj_mp = sp.coo_matrix(ep_st_obj)

            if has_action_repr:
                ep_st_obj_act = np.matmul(ep_st_obj, obj_action) > 0
                ep_st_obj_act_mp = sp.coo_matrix(ep_st_obj_act)
                ep_st_obj_aff = np.matmul(ep_st_obj_act, action_aff) > 0
                ep_st_obj_act_aff_mp = sp.coo_matrix(ep_st_obj_aff)



            ep_st_aff_st = np.matmul(ep_st_aff, aff_st) > 0
            ep_st_aff_st_mp = sp.coo_matrix(ep_st_aff_st)

            # st_obj_action = np.matmul(st_obj, obj_action) > 0
            # st_obj_action = sp.coo_matrix(st_obj_action)
            #
            # st_obj_action_aff = np.matmul(st_obj_action, action_aff) > 0
            # st_obj_action_aff_mp = sp.coo_matrix(st_obj_action_aff)
            #
            # ep_st_obj_action_aff = np.matmul(ep_state, st_obj_action_aff) > 0
            # ep_st_obj_action_aff_mp = sp.coo_matrix(ep_st_obj_action_aff)
            epstaffst_mp = self.sparse_mx_to_torch_sparse_tensor(ep_st_aff_st_mp)

            if has_action_repr:
                epstobjactaff_mp = self.sparse_mx_to_torch_sparse_tensor(ep_st_obj_act_aff_mp)
                return [epstobjactaff_mp, epstaffst_mp]
            else:
                epstobj_mp = self.sparse_mx_to_torch_sparse_tensor(ep_st_obj_mp)

            return [epstobj_mp, epstaffst_mp]
        else:
            return []

    def generate_mps_single_state(self, nodes, rels):
        import scipy.sparse as sp
        import torch

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data.astype(np.float32))
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse_coo_tensor(indices, values, shape).coalesce()

        state_keys = [n.element_id for n in nodes if 'StateT' in n.labels]
        obj_keys = [n.element_id for n in nodes if 'ObjectConcept' in n.labels]
        act_keys = [n.element_id for n in nodes if 'ActionRepr' in n.labels]
        aff_keys = [n.element_id for n in nodes if 'Affordance' in n.labels]

        idx_state = {id: i for i, id in enumerate(state_keys)}
        idx_obj = {id: i for i, id in enumerate(obj_keys)}
        idx_act = {id: i for i, id in enumerate(act_keys)}
        idx_aff = {id: i for i, id in enumerate(aff_keys)}

        state_obj_x, state_obj_y = [], []
        obj_act_x, obj_act_y = [], []
        act_aff_x, act_aff_y = [], []
        state_aff_x, state_aff_y = [], []

        for rel in rels:
            src, tgt = rel.start_node.element_id, rel.end_node.element_id
            if rel.type == "has_object":
                if src in idx_state and tgt in idx_obj:
                    state_obj_x.append(idx_state[src])
                    state_obj_y.append(idx_obj[tgt])
            elif rel.type == "contribute":
                if src in idx_obj and tgt in idx_act:
                    obj_act_x.append(idx_obj[src])
                    obj_act_y.append(idx_act[tgt])
            elif rel.type == "produces":
                if src in idx_state and tgt in idx_aff:
                    state_aff_x.append(idx_state[src])
                    state_aff_y.append(idx_aff[tgt])
            elif rel.type == "outcome":
                if src in idx_aff and tgt in idx_obj:
                    act_aff_x.append(idx_aff[src])
                    act_aff_y.append(idx_obj[tgt])

        num_states = len(state_keys)
        num_objs = len(obj_keys)
        num_acts = len(act_keys)
        num_affs = len(aff_keys)

        ep_state = sp.coo_matrix(([1.0], ([0], [0])), shape=(1, num_states)).toarray()
        st_obj = sp.coo_matrix((np.ones(len(state_obj_x)), (state_obj_x, state_obj_y)),
                               shape=(num_states, num_objs)).toarray()
        obj_act = sp.coo_matrix((np.ones(len(obj_act_x)), (obj_act_x, obj_act_y)), shape=(num_objs, num_acts)).toarray()
        act_aff = sp.coo_matrix((np.ones(len(act_aff_x)), (act_aff_x, act_aff_y)), shape=(num_affs, num_objs)).toarray()
        st_aff = sp.coo_matrix((np.ones(len(state_aff_x)), (state_aff_x, state_aff_y)),
                               shape=(num_states, num_affs)).toarray()

        ep_st_obj = (ep_state @ st_obj) > 0
        ep_st_obj_act = (ep_st_obj @ obj_act) > 0
        ep_st_aff = (ep_state @ st_aff) > 0

        return [
            sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ep_st_obj_act)),
            sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ep_st_aff)),
        ]

    def generate_mps_single_state_minigrid(self, nodes, rels):
        import scipy.sparse as sp
        import torch

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data.astype(np.float32))
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse_coo_tensor(indices, values, shape).coalesce()

        state_keys = [n.element_id for n in nodes if 'StateT' in n.labels]
        obj_keys = [n.element_id for n in nodes if 'ObjectConcept' in n.labels]
        aff_keys = [n.element_id for n in nodes if 'Affordance' in n.labels]

        idx_state = {id: i for i, id in enumerate(state_keys)}
        idx_obj = {id: i for i, id in enumerate(obj_keys)}
        idx_aff = {id: i for i, id in enumerate(aff_keys)}

        state_obj_x, state_obj_y = [], []
        obj_act_x, obj_act_y = [], []
        act_aff_x, act_aff_y = [], []
        state_aff_x, state_aff_y = [], []

        for rel in rels:
            src, tgt = rel.start_node.element_id, rel.end_node.element_id
            if rel.type == "has_object":
                if src in idx_state and tgt in idx_obj:
                    state_obj_x.append(idx_state[src])
                    state_obj_y.append(idx_obj[tgt])
            elif rel.type == "produces":
                if src in idx_state and tgt in idx_aff:
                    state_aff_x.append(idx_state[src])
                    state_aff_y.append(idx_aff[tgt])
            elif rel.type == "outcome":
                if src in idx_aff and tgt in idx_obj:
                    act_aff_x.append(idx_aff[src])
                    act_aff_y.append(idx_obj[tgt])

        num_states = len(state_keys)
        num_objs = len(obj_keys)
        num_affs = len(aff_keys)

        ep_state = sp.coo_matrix(([1.0], ([0], [0])), shape=(1, num_states)).toarray()
        st_obj = sp.coo_matrix((np.ones(len(state_obj_x)), (state_obj_x, state_obj_y)),
                               shape=(num_states, num_objs)).toarray()
        st_aff = sp.coo_matrix((np.ones(len(state_aff_x)), (state_aff_x, state_aff_y)),
                               shape=(num_states, num_affs)).toarray()

        ep_st_obj = (ep_state @ st_obj) > 0
        ep_st_aff = (ep_state @ st_aff) > 0

        return [
            sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ep_st_obj)),
            sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ep_st_aff)),
        ]

    def get_nei_index_st(self, file_name, obj_keys, aff_keys, data=None, full_path=None):
        state_nei_index = []
        if data is None:
            data = self.load_state_data(file_name, full_path=full_path)
        if len(data['StateT']) > 0:
            state_id = data['StateT'][0][0]
            objects_nei = [obj_keys.index(obj[1]) for obj in data['ObjectConceptRel'] if obj[0] == state_id]
            affs_nei = [aff_keys.index(aff[1]) for aff in data['AffordanceRel'] if aff[0] == state_id]
            state_nei_index.append(torch.tensor([objects_nei]).cuda())
            state_nei_index.append(torch.tensor([affs_nei]).cuda())
        return state_nei_index

    def get_nei_index_ep(self, file_name, state_keys, data=None, full_path=None):
        ep_nei_index = []
        if data is None:
            data = self.load_ep_data(file_name, full_path=full_path)
        if len(data['Episode']) > 0:
            ep_id = data['Episode'][0][0]
            states_nei = [state_keys.index(obj[1]) for obj in data['StateTRel'] if obj[0] == ep_id]
            ep_nei_index.append(torch.tensor([states_nei]).cuda())
        return ep_nei_index

    def fetch_data_for_ep_pickle(self, random_states_file, state_keys, data=None, full_path=None):
        ep_data = []
        keys = []
        ep_feats, full_ep = self.get_feats_episode(random_states_file, data=data, full_path=full_path)
        ep_nei = self.get_nei_index_ep(random_states_file, state_keys, data=data, full_path=full_path)
        ep_feats = ep_feats
        ep_data.append((ep_feats, ep_nei))
        return ep_data, full_ep

    def fetch_data_for_states_pickle(self, random_states_file, obj_keys, aff_keys, data=None, full_path=None):
        states_data = []
        keys = []
        state_feats, full_state = self.get_feats_state_st(random_states_file, data=data, full_path=full_path)
        state_nei = self.get_nei_index_st(random_states_file, obj_keys, aff_keys, data=data, full_path=full_path)
        state_feats = state_feats
        state_nei = state_nei
        states_data.append((state_feats, state_nei))
        return states_data, full_state

    def get_neighboring_values(self, n, lower_bound=0, upper_bound=30):
        neighbors = [
            max(n - 3, lower_bound),
            max(n - 2, lower_bound),
            min(n + 2, upper_bound),
            min(n + 3, upper_bound)
        ]
        return neighbors

    def choose_times_with_sufficient_similarity_dismi(self, disim_space=3):
        random_time_positive_1 = random.randint(0, 25)
        random_time_positive_2 = choice(self.get_neighboring_values(random_time_positive_1))
        random_time_negative = random_time_positive_2
        while random_time_negative == random_time_positive_2 or random_time_negative== random_time_positive_1:
            random_time_negative = choice(self.get_neighboring_values(max(random_time_positive_1, random_time_positive_2)+disim_space))

        result_query_sim, nr_results = self.cs_memory.check_similar_states_time_based(random_time_positive_1,
                                                                                  random_time_positive_2, sim_disim='sim')

        if nr_results == 0:
            return self.choose_times_with_sufficient_similarity_dismi()

        random_row_sim = result_query_sim.sample(n=1)
        anchor_id = random_row_sim['elementId(s)'].values[0]
        positive_id_2 = random_row_sim["elementId(s1)"].values[0]
        result_query_disim, nr_results_disim = self.cs_memory.check_similar_states_time_based(random_time_positive_1,
                                                                                  random_time_negative, sim_disim='disim', s_id=anchor_id)
        if nr_results_disim == 0:
            self.choose_times_with_sufficient_similarity_dismi()
        random_row_disim = result_query_disim.sample(n=1)
        negative_id = random_row_disim["elementId(s1)"].values[0]
        return anchor_id, positive_id_2, negative_id, random_time_positive_1, random_time_positive_2, random_time_negative

    def get_subgraph_state_data(self, batch_size=6):
        # #Fetching all nodes s.t to mark that some objects, affordances are shared between states
        # with open(f'{os.path.join(os.getcwd(), "states_data")}\\states_embeds.pickle',
        #           'rb') as handle:
        #     states_data = pickle.load(handle)

        # random_time_positive = random.randint(0, 30)
        # random_time_negative = choice([i for i in range(0, 30) if i not in [random_time_positive]])
        # random_time_positive_1, random_time_positive_2, random_time_negative = self.choose_times_with_sufficient_similarity_dismi()
        anchor_id, pos_id, neg_id, time_anchor, time_pos, time_neg = self.choose_times_with_sufficient_similarity_dismi()
        anchor_file = os.path.join(f'{os.path.join(os.getcwd(), "states_data_time_2")}\\', f"{time_anchor}")
        anchor_file_path = f"{anchor_file}\\{str(anchor_id).replace(':','-')}.pickle"

        pos_file = os.path.join(f'{os.path.join(os.getcwd(), "states_data_time_2")}\\', f"{time_pos}")
        pos_file_path = f"{pos_file}\\{str(pos_id).replace(':','-')}.pickle"

        neg_file = os.path.join(f'{os.path.join(os.getcwd(), "states_data_time_2")}\\', f"{time_neg}")
        neg_file_path = f"{neg_file}\\{str(neg_id).replace(':','-')}.pickle"


    def get_subgraph_episode_data(self, batch_size=2):
        task_1 = random.choice(self.tasks)
        task_2 = random.choice(self.tasks)
        while task_1 == task_2:
            task_2 = random.choice(self.tasks)

        succ_fail = random.choice(['true', 'false'])
        if task_1 != task_2:
            files_path_1 = os.path.join(f'{os.path.join(os.getcwd(), "ep_data")}\\', f"{task_1}", succ_fail)
            files_path_2 = os.path.join(f'{os.path.join(os.getcwd(), "ep_data")}\\', f"{task_2}", succ_fail)
            files_task_1 = os.listdir(files_path_1)
            files_task_2 = os.listdir(files_path_2)
            anchor_file = random.choice(files_task_1)
            positive_file = random.choice(files_task_1)
            negative_file = random.choice(files_task_2)

            state_keys, obj_keys, action_keys, aff_keys = self.get_all_keys()

            # with open(f'{os.path.join(os.getcwd(), "ep_data_emb")}\\states_embeds.pickle',
            #           'rb') as handle:
            #     state_data = pickle.load(handle)
            #     state_keys = list(state_data.keys())
            #
            # with open(f'{os.path.join(os.getcwd(), "ep_data_emb")}\\actions_embeds.pickle',
            #           'rb') as handle:
            #     action_data = pickle.load(handle)
            #     action_keys = list(action_data.keys())
            #
            # with open(f'{os.path.join(os.getcwd(), "ep_data_emb")}\\affs_embeds.pickle',
            #           'rb') as handle:
            #     affordance_data = pickle.load(handle)
            #     aff_keys = list(affordance_data.keys())
            #
            # with open(f'{os.path.join(os.getcwd(), "ep_data_emb")}\\objects_embeds.pickle',
            #           'rb') as handle:
            #     objects_data = pickle.load(handle)
            #     obj_keys = list(objects_data.keys())

            ep_data_1, full_ep_data = self.fetch_data_for_ep_pickle(os.path.join(files_path_1, anchor_file), state_keys, full_path=os.path.join(files_path_1, anchor_file))


            ep_data_2, full_ep_data_2 = self.fetch_data_for_ep_pickle(os.path.join(files_path_1, positive_file) , state_keys, full_path=os.path.join(files_path_1, positive_file))

            ep_data_3, full_ep_data_3 = self.fetch_data_for_ep_pickle(os.path.join(files_path_2, negative_file), state_keys,
                                                                              full_path=os.path.join(files_path_2, negative_file))

            keys = []
            return (ep_data_1, ep_data_2, ep_data_3), state_keys, aff_keys, obj_keys, action_keys, \
                (full_ep_data, full_ep_data_2, full_ep_data_3)

        return



    def get_subgraph_episode_data_minigrid(self, batch_size=2):
        task_1 = random.choice(self.tasks)
        task_2 = random.choice(self.tasks)
        while task_1 == task_2:
            task_2 = random.choice(self.tasks)

        succ_fail = random.choice(['true', 'false'])
        if task_1 != task_2:
            files_path_1 = os.path.join(f'{os.path.join(os.getcwd(), "ep_data_minigrid")}\\', f"{task_1}", succ_fail)
            files_path_2 = os.path.join(f'{os.path.join(os.getcwd(), "ep_data_minigrid")}\\', f"{task_2}", succ_fail)
            files_task_1 = os.listdir(files_path_1)
            files_task_2 = os.listdir(files_path_2)
            anchor_file = random.choice(files_task_1)
            positive_file = random.choice(files_task_1)
            negative_file = random.choice(files_task_2)

            state_keys, obj_keys, aff_keys = self.get_all_keys(has_action_repr=False)

            ep_data_1, full_ep_data = self.fetch_data_for_ep_pickle(os.path.join(files_path_1, anchor_file), state_keys, full_path=os.path.join(files_path_1, anchor_file))


            ep_data_2, full_ep_data_2 = self.fetch_data_for_ep_pickle(os.path.join(files_path_1, positive_file) , state_keys, full_path=os.path.join(files_path_1, positive_file))

            ep_data_3, full_ep_data_3 = self.fetch_data_for_ep_pickle(os.path.join(files_path_2, negative_file), state_keys,
                                                                              full_path=os.path.join(files_path_2, negative_file))

            keys = []
            return (ep_data_1, ep_data_2, ep_data_3), state_keys, aff_keys, obj_keys, \
                (full_ep_data, full_ep_data_2, full_ep_data_3)

        return






    def get_subgraph_episode_data_by_task_incp(self, task_1="CreateLevelPush-v0", succ_fail='true'):

        files_path_1 = os.path.join(f'{os.path.join(os.getcwd(),  "ep_data_incomplete_40")}\\', f"{task_1}", succ_fail)
        print(files_path_1)
        files_task_1 = os.listdir(files_path_1)
        anchor_file = random.choice(files_task_1)
        state_keys, obj_keys, action_keys, aff_keys = self.get_all_keys()
        ep_data_1, full_ep_data = self.fetch_data_for_ep_pickle(os.path.join(files_path_1, anchor_file), state_keys, full_path=os.path.join(files_path_1, anchor_file))

        return ep_data_1, state_keys, aff_keys, obj_keys, action_keys, full_ep_data, os.path.join(files_path_1, anchor_file)

    def get_subgraph_episode_data_by_task(self, task_1="CreateLevelPush-v0", succ_fail='true'):

        files_path_1 = os.path.join(f'{os.path.join(os.getcwd(), "ep_data")}\\', f"{task_1}", succ_fail)
        print(files_path_1)
        files_task_1 = os.listdir(files_path_1)
        anchor_file = random.choice(files_task_1)
        state_keys, obj_keys, action_keys, aff_keys = self.get_all_keys()
        ep_data_1, full_ep_data = self.fetch_data_for_ep_pickle(os.path.join(files_path_1, anchor_file), state_keys,
                                                                full_path=os.path.join(files_path_1, anchor_file))

        return ep_data_1, state_keys, aff_keys, obj_keys, action_keys, full_ep_data, os.path.join(files_path_1, anchor_file)

    def get_subgraph_episode_data_by_task_minigrid(self, task_1="MiniGrid-Empty-5x5-v0", succ_fail='true'):

        files_path_1 = os.path.join(f'{os.path.join(os.getcwd(), "ep_data_minigrid")}\\', f"{task_1}", succ_fail)
        print(files_path_1)
        files_task_1 = os.listdir(files_path_1)
        anchor_file = random.choice(files_task_1)
        state_keys, obj_keys, aff_keys =self.get_all_keys(has_action_repr=False)
        ep_data_1, full_ep_data = self.fetch_data_for_ep_pickle(os.path.join(files_path_1, anchor_file), state_keys,
                                                                full_path=os.path.join(files_path_1, anchor_file))

        return ep_data_1, state_keys, aff_keys, obj_keys, full_ep_data, os.path.join(files_path_1,
                                                                                                  anchor_file)

    def get_subgraph_episode_data_by_id(self, path):
        """
        Fetch subgraph episode data for a given list of episode IDs.

        Parameters:
        - episode_ids: List of episode IDs to fetch data for.
        - task: The task name associated with the episodes.
        - succ_fail: Specify if the episodes are successful ('true') or unsuccessful ('false').

        Returns:
        - episodes_data: A dictionary containing data for the requested episodes.
        """

        state_keys, obj_keys, action_keys, aff_keys = self.get_all_keys()
        ep_data_1, full_ep_data = self.fetch_data_for_ep_pickle(path, state_keys,
                                                                full_path=path)

        return ep_data_1, state_keys, aff_keys, obj_keys, action_keys, full_ep_data, path



    def get_episode_state_data(self):



        # pattern_positive_1 = re.compile(fr'states_data_t{random_time_positive_1}_\d+\.pickle')
        # pattern_positive_2 = re.compile(fr'states_data_t{random_time_positive_2}_\d+\.pickle')
        # pattern_negative = re.compile(fr'states_data_t{random_time_negative}_\d+\.pickle')

        # matching_files_positive_1 = [file for file in os.listdir(f'{os.path.join(os.getcwd(), "states_data")}\\') if pattern_positive_1.match(file)]
        # matching_files_positive_2 = [file for file in os.listdir(f'{os.path.join(os.getcwd(), "states_data")}\\') if
        #                            pattern_positive_2.match(file)]
        # matching_files_negative = [file for file in os.listdir(f'{os.path.join(os.getcwd(), "states_data")}\\') if pattern_negative.match(file)]
        #
        # if len(matching_files_positive_1) ==0 or len(matching_files_positive_2) == 0 or len(matching_files_negative) == 0:
        #     print(random_time_positive_1)
        #     print(random_time_positive_2)
        #     print(random_time_negative)
        #     return self.get_subgraph_state_data()
        #
        # random_states_file_pos_1 = random.choices(matching_files_positive_1, k=batch_size)[0]
        # random_states_file_pos_2 = random.choices(matching_files_positive_2, k=batch_size)[0]
        # random_states_file_neg_1 = random.choices(matching_files_negative, k=batch_size)[0]

        with open(f'{os.path.join(os.getcwd(), "states_data")}\\actions_embeds_2.pickle',
                  'rb') as handle:
            action_data = pickle.load(handle)
            action_keys = list(action_data.keys())


        with open(f'{os.path.join(os.getcwd(), "states_data")}\\affs_embeds_2.pickle',
                  'rb') as handle:
            affordance_data = pickle.load(handle)
            aff_keys = list(affordance_data.keys())

        with open(f'{os.path.join(os.getcwd(), "states_data")}\\objects_embeds_2.pickle',
                  'rb') as handle:
            objects_data = pickle.load(handle)
            obj_keys = list(objects_data.keys())

        states_data_p1, full_state_p1 = self.fetch_data_for_states_pickle(anchor_file_path, obj_keys, aff_keys, full_path=anchor_file_path)
        states_data_p2, full_state_p2 = self.fetch_data_for_states_pickle(pos_file_path, obj_keys, aff_keys, full_path=pos_file_path)
        states_data_n1, full_state_n1 = self.fetch_data_for_states_pickle(neg_file_path, obj_keys, aff_keys, full_path=neg_file_path)

        keys = []
        return (states_data_p1, states_data_p2, states_data_n1), keys, aff_keys, obj_keys, action_keys, \
               (full_state_p1, full_state_p2, full_state_n1), (time_anchor, time_pos, time_neg)

    def get_subgraph_state_data_no_sim_check(self,disim_space=3, batch_size=1):
        random_time_positive_1 = random.randint(0, 25)
        random_time_positive_2 = choice(self.get_neighboring_values(random_time_positive_1))
        random_time_negative = random_time_positive_2
        while random_time_negative == random_time_positive_2 or random_time_negative== random_time_positive_1:
            random_time_negative = choice(self.get_neighboring_values(max(random_time_positive_1, random_time_positive_2)+disim_space))

        matching_files_positive_1 = [file for file in os.listdir(f'{os.path.join(os.getcwd(), "states_data_time_2")}\\{random_time_positive_1}') ]
        matching_files_positive_2 = [file for file in os.listdir(f'{os.path.join(os.getcwd(), "states_data_time_2")}\\{random_time_positive_2}') ]
        matching_files_negative = [file for file in os.listdir(f'{os.path.join(os.getcwd(), "states_data_time_2")}\\{random_time_negative}')]



        random_states_file_pos_1 = random.choices(matching_files_positive_1, k=batch_size)[0]
        random_states_file_pos_2 = random.choices(matching_files_positive_2, k=batch_size)[0]
        random_states_file_neg_1 = random.choices(matching_files_negative, k=batch_size)[0]

        with open(f'{os.path.join(os.getcwd(), "states_data")}\\actions_embeds_2.pickle',
                  'rb') as handle:
            action_data = pickle.load(handle)
            action_keys = list(action_data.keys())

        with open(f'{os.path.join(os.getcwd(), "states_data")}\\affs_embeds_2.pickle',
                  'rb') as handle:
            affordance_data = pickle.load(handle)
            aff_keys = list(affordance_data.keys())

        with open(f'{os.path.join(os.getcwd(), "states_data")}\\objects_embeds_2.pickle',
                  'rb') as handle:
            objects_data = pickle.load(handle)
            obj_keys = list(objects_data.keys())

        states_data_p1, full_state_p1 = self.fetch_data_for_states_pickle(f'{os.path.join(os.getcwd(), "states_data_time_2")}\\{random_time_positive_1}\\{random_states_file_pos_1}', obj_keys, aff_keys,
                                                                          full_path=f'{os.path.join(os.getcwd(), "states_data_time_2")}\\{random_time_positive_1}\\{random_states_file_pos_1}')
        states_data_p2, full_state_p2 = self.fetch_data_for_states_pickle(f'{os.path.join(os.getcwd(), "states_data_time_2")}\\{random_time_positive_2}\\{random_states_file_pos_2}', obj_keys, aff_keys,
                                                                          full_path=f'{os.path.join(os.getcwd(), "states_data_time_2")}\\{random_time_positive_2}\\{random_states_file_pos_2}')
        states_data_n1, full_state_n1 = self.fetch_data_for_states_pickle(f'{os.path.join(os.getcwd(), "states_data_time_2")}\\{random_time_negative}\\{random_states_file_neg_1}', obj_keys, aff_keys,
                                                                          full_path=f'{os.path.join(os.getcwd(), "states_data_time_2")}\\{random_time_negative}\\{random_states_file_neg_1}')

        keys = []
        return (states_data_p1, states_data_p2, states_data_n1), keys, aff_keys, obj_keys, action_keys, \
               (full_state_p1, full_state_p2, full_state_n1), (0, 0, 0)



    def iterate_files_and_folders_for_feats(self, directory=os.path.join(os.path.dirname(os.getcwd()), "states_data_time_2")):
        states = {}
        objects = {}
        action_reprs = {}
        affordances = {}
        feats = []
        for parent_root, dirs, files_parent in os.walk(directory):
            print(f"Current directory: {parent_root}")
            for subdir in dirs:
                subdir_path = os.path.join(parent_root, subdir)
                print(f"Subdirectory: {subdir_path}")
                for root, dir, files in os.walk(subdir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        print(f"File: {file_path}")
                        data = self.load_state_data(None, full_path=file_path)
                        states_dict = dict(data['StateT'])
                        states.update(states_dict)
                        object_dict = dict(data['ObjectConcept'])
                        objects.update(object_dict)
                        affordance = dict(data['Affordance'])
                        affordances.update(affordance)
                        action_repr = dict(data['ActionRepr'])
                        action_reprs.update(action_repr)


            # Iterate through files
        feats.append(list(states.values()))
        feats.append(list(objects.values()))
        feats.append(list(action_reprs.values()))
        feats.append(list(affordances.values()))
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\states_embeds_2.pickle',
                  'wb') as handle:
            pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\objects_embeds_2.pickle',
                  'wb') as handle:
            pickle.dump(objects, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\actions_embeds_2.pickle',
                  'wb') as handle:
            pickle.dump(action_reprs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\affs_embeds_2.pickle',
                  'wb') as handle:
            pickle.dump(affordances, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "train_data_states")}\\feats_2.pickle',
                  'wb') as handle:
            pickle.dump(feats, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def iterate_files_and_folders_for_feats_ep(self, directory=os.path.join(os.path.dirname(os.getcwd()), "ep_data")):
        episodes = {}
        states = {}
        objects = {}
        action_reprs = {}
        affordances = {}
        feats = []
        for parent_root, dirs, files_parent in os.walk(directory):
            print(f"Current directory: {parent_root}")
            for subdir in dirs:
                subdir_path = os.path.join(parent_root, subdir)
                print(f"Subdirectory: {subdir_path}")
                for root, dir, files in os.walk(subdir_path):
                    if dir is not None:
                        for d in dir:
                            print(f"Sub sub dir:{d}")
                            sub_sub_dir_path = os.path.join(subdir_path,d)
                            for r,dd,fs in os.walk(sub_sub_dir_path):
                                for file in fs:
                                    file_path = os.path.join(r, file)
                                    print(f"File: {file_path}")
                                    data = self.load_ep_data(None, full_path=file_path)
                                    episodes_dict = dict(data['Episode'])
                                    episodes.update(episodes_dict)

                                    states_dict = dict(data['StateT'])
                                    states.update(states_dict)

                                    object_dict = dict(data['ObjectConcept'])
                                    objects.update(object_dict)

                                    affordance = dict(data['Affordance'])
                                    affordances.update(affordance)

                                    action_repr = dict(data['ActionRepr'])
                                    action_reprs.update(action_repr)

            # Iterate through files
        feats.append(list(states.values()))
        feats.append(list(objects.values()))
        feats.append(list(action_reprs.values()))
        feats.append(list(affordances.values()))
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "ep_data_emb")}\\states_embeds.pickle',
                  'wb') as handle:
            pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "ep_data_emb")}\\objects_embeds.pickle',
                  'wb') as handle:
            pickle.dump(objects, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "ep_data_emb")}\\actions_embeds.pickle',
                  'wb') as handle:
            pickle.dump(action_reprs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "ep_data_emb")}\\affs_embeds.pickle',
                  'wb') as handle:
            pickle.dump(affordances, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "ep_train_data")}\\feats.pickle',
                  'wb') as handle:
            pickle.dump(feats, handle, protocol=pickle.HIGHEST_PROTOCOL)




    def get_feats(self):
        states = {}
        objects = {}
        action_reprs = {}
        affordances = {}
        feats = []
        pattern = re.compile(fr'states_data_t\d+_\d+\.pickle')
        matching_files_positive = [file for file in os.listdir(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\') if pattern.match(file)]

        for i in matching_files_positive:
            data = self.load_state_data(i)
            states_dict = dict(data['StateT'])
            states.update(states_dict)
            object_dict = dict(data['ObjectConcept'])
            objects.update(object_dict)
            affordance = dict(data['Affordance'])
            affordances.update(affordance)
            action_repr = dict(data['ActionRepr'])
            action_reprs.update(action_repr)

        feats.append(list(states.values()))
        feats.append(list(objects.values()))
        feats.append(list(action_reprs.values()))
        feats.append(list(affordances.values()))
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\states_embeds.pickle',
                  'wb') as handle:
            pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\objects_embeds.pickle',
                  'wb') as handle:
            pickle.dump(objects, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\actions_embeds.pickle',
                  'wb') as handle:
            pickle.dump(action_reprs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\affs_embeds.pickle',
                  'wb') as handle:
            pickle.dump(affordances, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "train_data_states")}\\feats.pickle',
                  'wb') as handle:
            pickle.dump(feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_nei_index(self):
        nei_index = []

        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\states_embeds.pickle',
                  'rb') as handle:
            states_data = pickle.load(handle)
            states_keys = list(states_data.keys())

        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\affs_embeds.pickle',
                  'rb') as handle:
            affordance_data = pickle.load(handle)
            aff_keys = list(affordance_data.keys())

        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\objects_embeds.pickle',
                  'rb') as handle:
            objects_data = pickle.load(handle)
            obj_keys = list(objects_data.keys())
        states_objects_concepts = [[]] * len(states_keys)
        states_affordances = [[]] * len(states_keys)
        for i in range(0, 58):
            data = self.load_state_data(i)
            if len(data['StateT']) > 0:
                state_id = data['StateT'][0][0]
                state_index = states_keys.index(state_id)
                objects_nei = [obj_keys.index(obj[1]) for obj in data['ObjectConceptRel'] if obj[0] == state_id]
                affs_nei = [aff_keys.index(aff[1]) for aff in data['AffordanceRel'] if aff[0] == state_id]
                states_objects_concepts[state_index] = objects_nei
                states_affordances[state_index] = affs_nei
        nei_index.append(states_objects_concepts)
        nei_index.append(states_affordances)
        with open(f'{os.path.join(os.path.dirname(os.getcwd()), "train_data_states")}\\nei_index.pickle',
                  'wb') as handle:
            pickle.dump(nei_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return nei_index


    def delete_empty_dicts(self):
        pattern = re.compile(fr'states_data_t\d+_\d+\.pickle')
        matching_files_positive = [file for file in os.listdir(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\') if pattern.match(file)]
        for file in matching_files_positive:
            data = self.load_state_data(file)
            if all(not v for v in data.values()):
                file_path = f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\{file}'
                os.remove(file_path)
                print(file_path)


    def clean_up(self):
        pattern = re.compile(fr'states_data_reduced_\d+\.pickle')
        matching_files_positive = [file for file in os.listdir(f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\') if pattern.match(file)]
        for file in matching_files_positive:
            file_path = f'{os.path.join(os.path.dirname(os.getcwd()), "states_data")}\\{file}'
            os.remove(file_path)
            print(file_path)





def main():
    st_loader = StateLoader(nr_mps=2, mps=None, use_memory='ltm2')
    tasks = [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-SimpleCrossingS9N1-v0"
    ]
    for task in tasks:
        print(f'Collecting true episodes for task {task}')
        data = st_loader.get_episode_data_minigrid(task,'true')
        print(f'Collecting false episodes for task {task}')
        data= st_loader.get_episode_data_minigrid(task,'false')
    #
    # push_succ_1, all_state_keys, all_aff_keys, all_obj_keys, action_keys, fstate_push_succ_p1, anchor_file_push_succ_1 = st_loader.get_subgraph_episode_data_by_task_incp(task_1="CreateLevelPush-v0", succ_fail='true')
    # feats = push_succ_1[0][0]
    # nei_index = push_succ_1[0][1]
    # mps = st_loader.generate_mps_episode(nei_index, fstate_push_succ_p1)
    # print(mps)
#     tasks = ["CreateLevelPush-v0",
# "CreateLevelBuckets-v0",
# "CreateLevelBasket-v0",
# "CreateLevelBelt-v0",
# "CreateLevelObstacle-v0"]
#     for task in tasks:
#         data  = st_loader.reorder_states(task=task, succesfull="true")
#         data  = st_loader.reorder_states(task=task, succesfull="false")

    #st_loader.clean_up()
    # st_loader.get_state_data(time=16)
    # for i in range(0, 31):
    #     print(i)
    #     st_loader.get_state_data(time=i)
    # st_loader.get_episode_data()
    # st_loader.iterate_files_and_folders_for_feats_ep()
    # state_date = st_loader.load_state_data(0)
    # data_feats, nei_index = st_loader.load_train_data()
    # print(nei_index)

    return

if __name__ == '__main__':
    main()