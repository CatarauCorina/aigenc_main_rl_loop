import os
import torch
from hetgraph_gt_encoder.data_helpers.data_preparation import StateLoader
from hetgraph_gt_encoder.models.HeCo import HeCo
from hetgraph_gt_encoder.heco_params import heco_params

os.environ['NEO4J_BOLT_URL']='bolt://localhost:7687'
os.environ['NEO_PASS']='rl123456'
os.environ['NEO_USER']='neo4j'

class LTMInitliser():
    def __init__(self, use_memory='outcomesmall', has_action_repr=True, tasks=None, minigrid_mem=None):
        count_mps = 2
        self.embedding_mapping = {
            'ObjectConcept': 'value',
            'ActionRepr': 'value',
            'Affordance': 'outcome',
            'StateT': 'state_enc'
        }

        args = heco_params()
        st_loader = StateLoader(nr_mps=2, mps=None, use_memory=use_memory, tasks=tasks)
        if minigrid_mem is not None:
            st_loader_init = StateLoader(nr_mps=2, mps=None, use_memory=minigrid_mem, tasks=tasks)
            checkpoint = os.path.join(os.getcwd(), 'ep_hybrid_1.5_29_0.6739.pkl')
        else:
            st_loader_init = StateLoader(nr_mps=2, mps=None)
            checkpoint = os.path.join(os.getcwd(), 'ep_hybrid_1.5_14_0.14917626976966858.pkl')

        (batch_pos1, batch_pos2, batch_neg1), all_state_keys, all_aff_keys, all_obj_keys, (
            fstate_p1, fstate_p2, fstate_n1) = st_loader_init.get_subgraph_episode_data_minigrid(batch_size=1)
        feats = batch_pos1[0][0]
        nei_index = batch_pos1[0][1]
        mps = st_loader_init.generate_mps_episode(nei_index, fstate_p1, has_action_repr=has_action_repr)
        mps_dims = [mp.shape[1] for mp in mps]
        feats_dim_list = [i.shape[1] for i in batch_pos1[0][0]]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        minigird = not has_action_repr
        model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                     count_mps, args.sample_rate, args.nei_num, args.tau, args.lam, mps_dims, minigrid=minigird).to(device)

        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        self.model = model
        self.st_loader = st_loader
        self.alpha = 0.5
        self.loss_type = None

    # def process_state_node_data(self, node):
    #     """
    #     Converts Neo4j node data into usable tensor for encoding.
    #     """
    #     node_type = list(node.labels)[0]
    #     node_id = node.element_id
    #     embedding = torch.tensor(node['embedding'], dtype=torch.float32)
    #     return node_type, node_id, embedding

    def process_state_node_data(self, node):
        node_type = set(node.labels).pop()
        node_id = node.element_id
        try:
            node_embedding = node._properties[self.embedding_mapping[node_type]]
        except:
            node_embedding = None
        return node_type, node_id, node_embedding

    def generate_mps_from_edges(self, node_list, edge_list):
        """
        Simulates MPS structure from edges. Returns [adjacency_tensor] like what your encoder expects.
        """
        node_index = {nid: idx for idx, (nid, _) in enumerate(node_list)}
        edge_index = torch.tensor([
            [node_index[src], node_index[tgt]] for (src, tgt) in edge_list
            if src in node_index and tgt in node_index
        ], dtype=torch.long).T  # shape [2, num_edges]
        return [edge_index]

    def get_feats_episode(self, ep_data):
        feats = []

        # StateT
        states_dict = dict(ep_data['StateT'])  # List[Tuple[node_id, embedding]]
        feats.append(torch.stack([torch.tensor(v, dtype=torch.float32).cuda() for _, v in states_dict.items()]))

        # ObjectConcept
        object_dict = dict(ep_data['ObjectConcept'])
        if object_dict:
            feats.append(torch.stack([torch.tensor(v, dtype=torch.float32).cuda() for _, v in object_dict.items()]))
        else:
            feats.append(torch.empty(0).cuda())

        # Affordance
        aff_dict = dict(ep_data['Affordance'])
        aff_vals = [v for _, v in aff_dict.items() if v is not None]
        if aff_vals:
            feats.append(torch.stack([torch.tensor(v, dtype=torch.float32).cuda() for v in aff_vals]))
        else:
            feats.append(torch.empty(0).cuda())

        # ActionRepr
        action_dict = dict(ep_data['ActionRepr'])
        if action_dict:
            feats.append(torch.stack([torch.tensor(v, dtype=torch.float32).cuda() for _, v in action_dict.items()]))
        else:
            feats.append(torch.empty(0).cuda())

        return feats, ep_data

    def get_feats_episode_minigrid(self, ep_data):
        feats = []

        # StateT
        states_dict = dict(ep_data['StateT'])  # List[Tuple[node_id, embedding]]
        feats.append(torch.stack([torch.tensor(v, dtype=torch.float32).cuda() for _, v in states_dict.items()]))

        # ObjectConcept
        object_dict = dict(ep_data['ObjectConcept'])
        if object_dict:
            feats.append(torch.stack([torch.tensor(v, dtype=torch.float32).cuda() for _, v in object_dict.items()]))
        else:
            feats.append(torch.empty(0).cuda())

        # Affordance
        aff_dict = dict(ep_data['Affordance'])
        aff_vals = [v for _, v in aff_dict.items() if v is not None]
        if aff_vals:
            feats.append(torch.stack([torch.tensor(v, dtype=torch.float32).cuda() for v in aff_vals]))
        else:
            feats.append(torch.empty(0).cuda())

        return feats, ep_data

    def get_nei_index_ep(self, state_keys, ep_data):
        """
        Args:
            state_keys: List of all state node IDs in order (used as index)
            ep_data: Dictionary for the episode (already parsed from Neo4j)

        Returns:
            ep_nei_index: List of torch tensors (neighbor index list for each episode)
        """

        ep_nei_index = []

        if len(ep_data['Episode']) > 0:
            ep_id = ep_data['Episode'][0][0]
            states_nei = [state_keys.index(obj[1]) for obj in ep_data['StateTRel'] if obj[0] == ep_id]
            ep_nei_index.append(torch.tensor([states_nei]).cuda())
        return ep_nei_index

    # def fetch_data_for_ep_pickle(self, random_states_file, state_keys, data=None, full_path=None):
    #     ep_data = []
    #     keys = []
    #     ep_feats, full_ep = self.get_feats_episode(random_states_file, data=data, full_path=full_path)
    #     ep_nei = self.get_nei_index_ep(random_states_file, state_keys, data=data, full_path=full_path)
    #     ep_feats = ep_feats
    #     ep_data.append((ep_feats, ep_nei))
    #     return ep_data, full_ep


    def encode_episode(self, episode_id):
        ep_id = episode_id
        print(f"🔄 Encoding episode {ep_id}...")

        try:
            ep_nodes, ep_rel = self.st_loader.cs_memory.get_episode_graph(ep_id)
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

            for node in ep_nodes:
                node_type, node_id, node_emb = self.process_state_node_data(node)
                ep_data[node_type].append((node_id, node_emb))

            if len(ep_data['StateT']) < 2:
                print(f"⚠️ Skipping episode {ep_id}: too few states.")
                return None

            state_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                          set(rel.end_node.labels).pop() == 'StateT']

            object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                           set(rel.end_node.labels).pop() == 'ObjectConcept']
            action_repr_rel = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                               set(rel.end_node.labels).pop() == 'ActionRepr']

            aff_rels_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                            set(rel.end_node.labels).pop() == 'Affordance']

            ep_data['ObjectConceptRel'] = object_rels
            ep_data['ActionReprRel'] = action_repr_rel
            ep_data['AffordanceRel'] = aff_rels_red
            ep_data['StateTRel'] = state_rels

            id = str(ep_id).replace(':', '-')

            all_state_keys, all_obj_keys, action_keys, all_aff_keys = self.st_loader.get_all_keys()
            feats, ep_data = self.get_feats_episode(ep_data)
            nei_index = self.get_nei_index_ep(all_state_keys, ep_data)
            # ep_data.append((feats, nei_index))

            mps = self.st_loader.generate_mps_episode(nei_index, ep_data)

            z_sc, _, _, _ = self.model(feats, mps, nei_index, self.alpha, self.loss_type, testing=True)
            # z = normalize(z, dim=0)  # optional but helpful
            return z_sc
        except Exception as e:
            print(f"❌ Failed to encode episode {ep_id}: {e}")
            return None


    def encode_episode_minigird(self, episode_id):
        ep_id = episode_id
        print(f"🔄 Encoding episode {ep_id}...")

        try:
            ep_nodes, ep_rel = self.st_loader.cs_memory.get_episode_graph_minigrid(ep_id)
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

            for node in ep_nodes:
                node_type, node_id, node_emb = self.process_state_node_data(node)
                ep_data[node_type].append((node_id, node_emb))

            if len(ep_data['StateT']) < 2:
                print(f"⚠️ Skipping episode {ep_id}: too few states.")
                return None

            state_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                          set(rel.end_node.labels).pop() == 'StateT']

            object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                           set(rel.end_node.labels).pop() == 'ObjectConcept']

            aff_rels_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
                            set(rel.end_node.labels).pop() == 'Affordance']

            ep_data['ObjectConceptRel'] = object_rels
            ep_data['AffordanceRel'] = aff_rels_red
            ep_data['StateTRel'] = state_rels

            id = str(ep_id).replace(':', '-')

            all_state_keys, all_obj_keys, all_aff_keys = self.st_loader.get_all_keys(has_action_repr=False)
            feats, ep_data = self.get_feats_episode_minigrid(ep_data)
            nei_index = self.get_nei_index_ep(all_state_keys, ep_data)
            # ep_data.append((feats, nei_index))

            mps = self.st_loader.generate_mps_episode(nei_index, ep_data, has_action_repr=False)

            z_sc, _, _, _ = self.model(feats, mps, nei_index, self.alpha, self.loss_type, testing=True)
            # z = normalize(z, dim=0)  # optional but helpful
            return z_sc
        except Exception as e:
            print(f"❌ Failed to encode episode {ep_id}: {e}")
            return None


    def encode_single_state_ep(self, state_id):
        print(f"🔄 Encoding single state {state_id}...")

        try:
            nodes, rels = self.st_loader.cs_memory.get_state_graph_setle(state_id)
            ep_data = {
                'Episode': [],  # empty for single state
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

            for node in nodes:
                node_type, node_id, node_emb = self.process_state_node_data(node)
                ep_data[node_type].append((node_id, node_emb))

            # Reconstruct relations
            state_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in rels if
                          set(rel.end_node.labels).pop() == 'StateT']
            object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in rels if
                           set(rel.end_node.labels).pop() == 'ObjectConcept']
            action_repr_rel = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in rels if
                               set(rel.end_node.labels).pop() == 'ActionRepr']
            affordance_rels = [
                (rel.nodes[0].element_id, rel.nodes[1].element_id)
                for rel in rels
                if 'Affordance' in rel.nodes[0].labels or 'Affordance' in rel.nodes[1].labels
            ]

            ep_data['StateTRel'] = state_rels
            ep_data['ObjectConceptRel'] = object_rels
            ep_data['ActionReprRel'] = action_repr_rel
            ep_data['AffordanceRel'] = affordance_rels

            # Prepare feature matrices
            all_state_keys, all_obj_keys, action_keys, all_aff_keys = self.st_loader.get_all_keys()
            feats, ep_data = self.get_feats_episode(ep_data)
            nei_index = self.get_nei_index_ep(all_state_keys, ep_data)
            mps = self.st_loader.generate_mps_single_state(nodes, rels)

            z_sc, _, _, _ = self.model(feats, mps, nei_index, self.alpha, self.loss_type, testing=True)
            return z_sc

        except Exception as e:
            print(f"❌ Failed to encode state {state_id}: {e}")
            return None

    def encode_single_state_ep_minigrid(self, state_id):
        print(f"🔄 Encoding single state {state_id}...")

        try:
            nodes, rels = self.st_loader.cs_memory.get_state_graph_setle_minigrid(state_id)
            ep_data = {
                'Episode': [],  # empty for single state
                'StateT': [],
                'ObjectConcept': [],
                'Affordance': [],
                "StateTRel": [],
                'ObjectConceptRel': [],
                'AffordanceRel': [],
                'full_rels': None
            }

            for node in nodes:
                node_type, node_id, node_emb = self.process_state_node_data(node)
                ep_data[node_type].append((node_id, node_emb))

            # ✅ Reconstruct relations
            state_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in rels if
                          set(rel.end_node.labels).pop() == 'StateT']
            object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in rels if
                           set(rel.end_node.labels).pop() == 'ObjectConcept']
            affordance_rels = [
                (rel.nodes[0].element_id, rel.nodes[1].element_id)
                for rel in rels
                if 'Affordance' in rel.nodes[0].labels or 'Affordance' in rel.nodes[1].labels
            ]

            ep_data['StateTRel'] = state_rels
            ep_data['ObjectConceptRel'] = object_rels
            ep_data['AffordanceRel'] = affordance_rels

            # ✅ Prepare feature matrices
            all_state_keys, all_obj_keys, all_aff_keys = self.st_loader.get_all_keys(has_action_repr=False)
            feats, ep_data = self.get_feats_episode_minigrid(ep_data)
            nei_index = self.get_nei_index_ep(all_state_keys, ep_data)
            mps = self.st_loader.generate_mps_single_state_minigrid(nodes, rels)

            # ✅ Run model (same as before)
            z_sc, _, _, _ = self.model(feats, mps, nei_index, self.alpha, self.loss_type, testing=True)
            return z_sc

        except Exception as e:
            print(f"❌ Failed to encode state {state_id}: {e}")
            return None

    def encode_all_episodes_minigrid(self, task=None, succesfull=None, partial_trace=False):
        all_ids = self.st_loader.cs_memory.get_episode_ids(task=task, succesful=succesfull)
        for index, ep_id_row in all_ids.iterrows():

            ep_id = ep_id_row['elementId(e)']
            z_sc = self.encode_episode_minigird(ep_id)
            print(f"🔄 Encoding episode {ep_id}...")
            if partial_trace:
                return z_sc
            if z_sc is not None: # Store encoded vector back to episode node
                self.st_loader.cs_memory.set_property(ep_id, "Episode", "set_embedding", z_sc.tolist()[0])
                print(f"✅ Stored embedding for episode {ep_id}")

    def encode_all_episodes(self, task=None, succesfull=None, partial_trace=False):
        all_ids = self.st_loader.cs_memory.get_episode_ids(task=task, succesful=succesfull)
        for index, ep_id_row in all_ids.iterrows():

            ep_id = ep_id_row['elementId(e)']
            z_sc = self.encode_episode(ep_id)
            print(f"🔄 Encoding episode {ep_id}...")
            if partial_trace:
                return z_sc
            if z_sc is not None: # Store encoded vector back to episode node
                self.st_loader.cs_memory.set_property(ep_id, "Episode", "set_embedding", z_sc.tolist()[0])
                print(f"✅ Stored embedding for episode {ep_id}")

            #
            # try:
            #     ep_nodes, ep_rel = self.st_loader.cs_memory.get_episode_graph(ep_id)
            #     ep_data = {
            #         'Episode': [],
            #         'StateT': [],
            #         'ObjectConcept': [],
            #         'Affordance': [],
            #         'ActionRepr': [],
            #         "StateTRel": [],
            #         'ObjectConceptRel': [],
            #         'ActionReprRel': [],
            #         'AffordanceRel': [],
            #         'full_rels': None
            #     }
            #
            #     for node in ep_nodes:
            #         node_type, node_id, node_emb = self.process_state_node_data(node)
            #         ep_data[node_type].append((node_id, node_emb))
            #
            #     if len(ep_data['StateT']) < 2:
            #         print(f"⚠️ Skipping episode {ep_id}: too few states.")
            #         continue
            #
            #     state_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
            #                   set(rel.end_node.labels).pop() == 'StateT']
            #     object_rels = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
            #                    set(rel.end_node.labels).pop() == 'ObjectConcept']
            #     action_repr_rel = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
            #                        set(rel.end_node.labels).pop() == 'ActionRepr']
            #
            #     aff_rels_red = [(rel.nodes[0].element_id, rel.nodes[1].element_id) for rel in ep_rel if
            #                     set(rel.end_node.labels).pop() == 'Affordance']
            #
            #     ep_data['ObjectConceptRel'] = object_rels
            #     ep_data['ActionReprRel'] = action_repr_rel
            #     ep_data['AffordanceRel'] = aff_rels_red
            #     ep_data['StateTRel'] = state_rels
            #
            #     id = str(ep_id).replace(':', '-')
            #
            #     all_state_keys, all_obj_keys, action_keys, all_aff_keys = self.st_loader.get_all_keys()
            #     feats, ep_data = self.get_feats_episode(ep_data)
            #     nei_index = self.get_nei_index_ep(all_state_keys, ep_data)
            #     # ep_data.append((feats, nei_index))
            #
            #     mps = self.st_loader.generate_mps_episode(nei_index, ep_data)
            #
            #     z_sc, _,_,_ = self.model(feats, mps, nei_index, self.alpha, self.loss_type, testing=True)
            #     # z = normalize(z, dim=0)  # optional but helpful
            #
            #     # Store encoded vector back to episode node
            #     self.st_loader.cs_memory.set_property(ep_id, "Episode", "set_embedding", z_sc.tolist()[0])
            #     print(f"✅ Stored embedding for episode {ep_id}")
            #
            # except Exception as e:
            #     print(f"❌ Failed to encode episode {ep_id}: {e}")

    def get_data_for_episode(self, batch, full_state):
        feats = batch[0][0]
        nei_index = batch[0][1]
        mps = self.st_loader.generate_mps_episode(nei_index, full_state)
        return feats, nei_index, mps


def main():
    tasks = [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-SimpleCrossingS9N1-v0"
    ]
    # ltm_process = LTMInitliser()
    ltm_process = LTMInitliser(use_memory='ltm2', has_action_repr=False, tasks=tasks, minigrid_mem='ltm2')

    ltm_process.encode_all_episodes_minigrid()
    return

if __name__ == '__main__':
    main()