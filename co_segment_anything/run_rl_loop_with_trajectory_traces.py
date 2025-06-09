import random
import wandb
import gym
import math
from collections import defaultdict

import sys
import json
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
from hetgraph_gt_encoder.encode_ltm import LTMInitliser
import torch
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True


IS_SERVER = True
from baseline_models.logger import Logger
from co_segment_anything.dqn_sam import DQN, DQNSetle
from co_segment_anything.sam_utils import SegmentAnythingObjectExtractor
from create.create_game.settings import CreateGameSettings
from memory_graph.gds_concept_space import ConceptSpaceGDS
from memory_graph.memory_utils import WorkingMemory
from affordance_learning.action_observation.utils import ActionObservation
import torch
import numpy as np
from torch.nn.functional import cosine_similarity
import torch
from hetgraph_gt_encoder.data_helpers.data_preparation import StateLoader
from hetgraph_gt_encoder.models.HeCo import HeCo
from hetgraph_gt_encoder.heco_params import heco_params
envs_to_run = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os

os.environ['NEO4J_BOLT_URL']='bolt://localhost:7687'
os.environ['NEO_PASS']='rl123456'
os.environ['NEO_USER']='neo4j'


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','mask', 'inventory'))

class EnrichedStateAdapter(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, enriched_embedding):
        return self.adapter(enriched_embedding)


class EnrichmentAttention(nn.Module):

    def __init__(self, query_dim, candidate_dim, proj_dim=128):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, proj_dim)
        self.k_proj = nn.Linear(candidate_dim, proj_dim)
        self.temp = nn.Parameter(torch.tensor(1.0))

    # def forward(self, query, candidates):
    #     """
    #     query: Tensor of shape [D] or [1, D]
    #     candidates: Tensor of shape [N, D]
    #     returns: Tensor of shape [N], the attention weights
    #     """
    #     if query.dim() == 1:
    #         query = query.unsqueeze(0)  # [1, D]
    #
    #     q = self.q_proj(query)  # [1, D]
    #     keys = self.k_proj(candidates)  # [N, D]
    #
    #     # Dot product attention: (N, D) x (D, 1) → (N, 1) → (N)
    #     attn_logits = torch.matmul(keys, q.T).squeeze(1) / np.sqrt(q.shape[-1])
    #     attn_weights = F.softmax(attn_logits, dim=0)  # [N]
    #
    #     return attn_weights

    def forward(self, query, candidates):
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, D_q]

        q = self.q_proj(query)  # [1, proj_dim]
        k = self.k_proj(candidates)  # [N, proj_dim]

        logits = (k @ q.T).squeeze(-1) / (self.temp * np.sqrt(q.shape[-1]))  # [N]
        return F.softmax(logits, dim=0)  # [N]


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_td(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        return state_batch, action_batch, reward_batch, non_final_next_states, torch.tensor(batch.done,
                                                                                            dtype=torch.int64), non_final_mask

    def __len__(self):
        return len(self.memory)




class SetleEnricher:
    def __init__(self, encoder, wm, ltm):
        self.embedding_mapping = {
            'ObjectConcept': 'value',
            'ActionRepr': 'value',
            'Affordance': 'outcome',
            'StateT': 'state_enc'
        }
        self.encoder = encoder  # SETLE encoder (e.g., HeCo)
        self.wm = wm            # WorkingMemory instance
        self.ltm_init = LTMInitliser(use_memory='workingmemory')
        self.match_tracking_table = wandb.Table(columns=["Timestep", "ShortID", "Frequency"])

        self.ltm = ltm
        self.match_counts = defaultdict(int)  # track how often each episode is matched

        self.attn_modules = {
            "Affordance": EnrichmentAttention(query_dim=64, candidate_dim=513).to(device),
            "ObjectConcept": EnrichmentAttention(query_dim=64, candidate_dim=512).to(device),
            "ActionRepr": EnrichmentAttention(query_dim=64, candidate_dim=16).to(device),
        }



    def encode_graph(self, graph):
        feats, mps = graph['feats'], graph['mps']
        return self.encoder(feats, mps)

    def find_closest_state_idx(self, query_z, matched_traj):
        similarities = []
        for g in matched_traj:
            z_g = self.encode_graph(g)
            sim = cosine_similarity(query_z, z_g, dim=0).item()
            similarities.append(sim)
        return int(np.argmax(similarities))

    def extract_partial_trace(self, episode_graph, window_size=4):
        """
        Convert last `window_size` graphs from an episode into a subgraph for SETLE encoding.
        """
        feats = [g['feats'] for g in episode_graph[-window_size:]]
        mps = [g['mps'] for g in episode_graph[-window_size:]]
        return {
            'feats': torch.cat(feats, dim=0),
            'mps': [torch.cat(m, dim=0) for m in zip(*mps)]
        }

    def enrich_graph_and_store(self, G_t_window, matched_traj, episode_id, state_id, reward_threshold=0):
        z_query = self.encode_graph(G_t_window)
        match_idx = self.find_closest_state_idx(z_query, matched_traj)
        future_graphs = matched_traj[match_idx + 1: match_idx + 4]

        for future_state in future_graphs:
            for node in future_state['nodes']:
                if 'affordance' in node and node.get('reward', 0) >= reward_threshold:
                    self.wm.concept_space.add_affordance_to_state(
                        state_id,
                        affordance=node['affordance'],
                        tool=node.get('tool'),
                        target=node['object'],
                        source="setle"
                    )

            for edge in future_state.get('edges', []):
                if edge.get('reward', 0) >= reward_threshold:
                    self.wm.concept_space.add_edge(
                        source=edge['from'],
                        target=edge['to'],
                        label=edge['type'],
                        weight=edge.get('reward', 1.0),
                        state_id=state_id,
                        source_tag="setle"
                    )

        self.wm.concept_space.tag_state(state_id, label="enhanced_with_setle")

    # def apply_attention_enrichment(self, query_embedding, candidates, full_nodes, attn_module, state_id, node_type,
    #                                writer, step, top_k=5):
    #     if not candidates:
    #         return
    #
    #     node_ids, embeddings = zip(*candidates)  # make sure candidates includes full_nodes
    #     embeddings = torch.stack(embeddings).to(query_embedding.device)
    #
    #     attn_weights = attn_module(query_embedding, embeddings)  # shape: [N]
    #     top_indices = torch.topk(attn_weights, k=min(top_k, len(attn_weights))).indices.tolist()
    #
    #     for idx in top_indices:
    #         node = full_nodes[idx]
    #         label = list(node.labels)[0]
    #         props = dict(node._properties)
    #         original_node_id = node.element_id
    #
    #         # 🧱 Create node in working memory
    #         new_node_id = self.wm.concept_space.create_node(label, props, source="setle_attention")
    #         rel_type = {
    #             "Affordance": "produces",
    #             "ObjectConcept": "has_object",
    #             "ActionRepr": "has_action"
    #         }.get(label, "connected_to")
    #
    #         if label == "ObjectConcept" or label == 'Affordance':
    #             self.wm.concept_space.add_edge(
    #                 source=state_id,
    #                 target=new_node_id,
    #                 label=rel_type,
    #                 weight=1.0,
    #                 state_id=state_id,
    #                 source_tag="setle_enrich"
    #             )
    #
    #
    #         elif label == "ActionRepr":
    #             # Check if similar ActionRepr node exists in WM
    #             matched = self.wm.concept_space.find_similar_action_repr(props)
    #             if matched:
    #                 for node_id in matched:
    #                     self.wm.concept_space.add_edge(
    #                         source=new_node_id,
    #                         target=node_id,
    #                         label="related_to",
    #                         weight=1.0,
    #                         state_id=state_id,
    #                         source_tag="setle_enrich"
    #                     )
    #             else:
    #                 # Fetch objects connected to ActionRepr in LTM and recreate them in WM
    #                 connected = self.ltm.concept_space.get_connected_nodes(original_node_id, edge_type="uses")
    #                 for conn_node in connected:
    #                     obj_props = dict(conn_node._properties)
    #                     obj_label = list(conn_node.labels)[0]
    #                     obj_id_new = self.wm.concept_space.create_node(obj_label, obj_props, source="setle_enrich")
    #
    #                     self.wm.concept_space.add_edge(
    #                         source=state_id,
    #                         target=obj_id_new,
    #                         label="has_object",
    #                         weight=1.0,
    #                         state_id=state_id,
    #                         source_tag="setle_enrich"
    #                     )
    #
    #                     self.wm.concept_space.add_edge(
    #                         source=new_node_id,
    #                         target=obj_id_new,
    #                         label="uses",
    #                         weight=1.0,
    #                         state_id=state_id,
    #                         source_tag="setle_enrich"
    #                     )
    #
    #             # Finally link ActionRepr to the state
    #             self.wm.concept_space.add_edge(
    #                 source=state_id,
    #                 target=new_node_id,
    #                 label="has_action",
    #                 weight=1.0,
    #                 state_id=state_id,
    #                 source_tag="setle_enrich"
    #             )
    #
    #     # 🧾 Log attention for debugging
    #     if writer is not None:
    #         table_data = [
    #             [node_ids[i], float(attn_weights[i].item())]
    #             for i in top_indices
    #         ]
    #         table = wandb.Table(data=table_data, columns=["Node ID", "Attention Weight"])
    #         writer.log({f"Attention/{node_type}_step_{step}": table})

    def apply_attention_enrichment(self, query_embedding, candidates, full_nodes, attn_module, state_id, node_type,
                                   writer, step, top_k=2):
        if not candidates:
            return

        node_ids, embeddings = zip(*candidates)
        embeddings = torch.stack(embeddings).to(query_embedding.device)
        print(query_embedding.shape)
        print(embeddings.shape)
        attn_weights = attn_module(query_embedding, embeddings)

        top_indices = torch.topk(attn_weights, k=min(top_k, len(attn_weights))).indices.tolist()

        for idx in top_indices:
            node = full_nodes[idx]
            label = list(node.labels)[0]
            props = dict(node._properties)
            original_node_id = node.element_id

            # 🧱 Create node in WM
            new_node_id = self.wm.concept_space.create_node(label, props, source="setle_attention")

            if label in ["ObjectConcept", "Affordance"]:
                rel_type = "has_object" if label == "ObjectConcept" else "produces"
                self.wm.concept_space.add_edge(
                    source=state_id,
                    target=new_node_id,
                    label=rel_type,
                    weight=1.0,
                    state_id=state_id,
                    source_tag="setle_enrich"
                )

            elif label == "ActionRepr":
                # 🔁 Fetch connected objects in LTM
                connected_objs = self.ltm.concept_space.get_connected_nodes(original_node_id, edge_type="contribute")
                for obj_node in connected_objs:
                    obj_props = dict(obj_node._properties)
                    obj_label = list(obj_node.labels)[0]
                    obj_emb = obj_props.get("value")

                    # ✅ Check if similar object already exists in WM
                    similar_obj_ids = self.wm.concept_space.find_similar_object_concepts(obj_emb)
                    if similar_obj_ids:
                        for sid in similar_obj_ids:
                            self.wm.concept_space.add_edge(
                                source=new_node_id,
                                target=sid,
                                label="contribute",
                                weight=1.0,
                                state_id=state_id,
                                source_tag="setle_enrich"
                            )
                    else:
                        # 🆕 Copy object to WM and link
                        obj_new_id = self.wm.concept_space.create_node(obj_label, obj_props, source="setle_enrich")
                        self.wm.concept_space.add_edge(
                            source=state_id,
                            target=obj_new_id,
                            label="has_object",
                            weight=1.0,
                            state_id=state_id,
                            source_tag="setle_enrich"
                        )
                        self.wm.concept_space.add_edge(
                            source=new_node_id,
                            target=obj_new_id,
                            label="contribute",
                            weight=1.0,
                            state_id=state_id,
                            source_tag="setle_enrich"
                        )

        # 📊 Log attention scores
        # if writer is not None:
        #     # try:
        #     #     self.log_attention_heatmap(attn_weights, f"{node_type} attention", writer, step)
        #     # except Exception as e:
        #     #     print("Heatmap error")
        #
        #     table_data = [[node_ids[i], float(attn_weights[i].item())] for i in top_indices]
        #     table = wandb.Table(data=table_data, columns=["Node ID", "Attention Weight"])
        #     writer.log({f"Attention/{node_type}_step_{step}": table})



    # def log_attention_heatmap(self, weights, title, writer, step):
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     fig, ax = plt.subplots()
    #     weights_np = weights.detach().cpu().numpy()
    #     sns.heatmap(weights_np, annot=True, cbar=False, ax=ax)
    #
    #     sns.heatmap(weights.unsqueeze(0).cpu().numpy(), annot=True, cbar=False, ax=ax)
    #     ax.set_title(title)
    #     writer.log({f"{title}_heatmap_step_{step}": wandb.Image(fig)})
    #     plt.close()

    def match_episodes(self, episode_id, writer, top_k=5, timestep=0, use_penalty=True):
        # Step 1: extract recent subgraph from the partial trace and encode it
        G_t_set_encoding = self.ltm_init.encode_episode(episode_id)

        if G_t_set_encoding is None:
            print(f"⚠️ Could not encode episode {episode_id}")
            return None, None

        # Step 2: Fetch all stored episode embeddings from LTM (Neo4j)
        stored_eps = self.ltm.concept_space.get_all_episode_embeddings()
        stored_eps_task = self.ltm.concept_space.get_all_episode_tasks()
        current_task = self.wm.concept_space.get_crt_episode_task(episode_id)


        # dict: {ep_id: torch.Tensor}
        if not stored_eps:
            print("⚠️ No episode embeddings found in LTM.")
            return None , None

        all_ids = list(stored_eps.keys())
        all_embeddings = torch.stack([stored_eps[eid] for eid in all_ids])

        # Step 3: Compute cosine similarity to all stored episodes
        sims = cosine_similarity(G_t_set_encoding, all_embeddings).squeeze(0)
        # 🔻 Penalize frequent matches (log-based decay or linear)
        if use_penalty:
            penalty = torch.tensor([
                1.0 / (1 + self.match_counts[eid]) for eid in all_ids
            ], device=sims.device)

            adjusted_sims = sims * penalty

            top_indices = torch.topk(adjusted_sims, k=min(top_k, len(all_ids))).indices.tolist()
            top_episode_ids = [all_ids[i] for i in top_indices]
            cross_matches = [
                eid for eid in top_episode_ids
                if stored_eps_task.get(eid, current_task) != current_task
            ]
            cross_task_ratio = len(cross_matches) / len(top_episode_ids)

            # writer.log({f"Generalization/CrossTaskMatchRatio_t{timestep}": cross_task_ratio})

            for eid in top_episode_ids:
                self.match_counts[eid] += 1  # update match count

            print(f"🔎 Top {top_k} matched episodes for {episode_id} at t={timestep}: {top_episode_ids}")


            top_matches = sorted(self.match_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

            # for eid, count in top_matches:
            #     short_id = eid.split(":")[-1]
            #     self.match_tracking_table.add_data(timestep, short_id, count)
            #
            # # Log line chart of matches over time
            # match_line = wandb.plot.line(self.match_tracking_table, "Timestep", "Frequency", "ShortID",
            #                              title="Top Episode Matches Over Time")
            # writer.log({f"Match/LineOverTime": match_line})

            # ✅ Log match rank with histogram instead of a table
            # match_scores = [adjusted_sims[i].item() for i in top_indices]
            # 🟩 Log top matched episodes as a bar chart (short IDs only)
            top_n = 10  # show top 10 most matched
            top_match_counts = sorted(self.match_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

            match_freq_table = wandb.Table(columns=["Episode ID", "Count"])
            # for eid, count in top_match_counts:
            #     short_id = eid.split(":")[-1]  # keep only the last part of the elementId
            #     match_freq_table.add_data(short_id, count)
            #
            # bar_chart = wandb.plot.bar(match_freq_table, "Episode ID", "Count",
            #                            title=f"Match Frequency (Top {top_n}) - t={timestep}")
            # writer.log({f"Match/FrequencyBar_Top{top_n}_t{timestep}": bar_chart})


        else:
            top_indices = torch.topk(sims, k=min(top_k, len(all_ids))).indices.tolist()

            top_episode_ids = [all_ids[i] for i in top_indices]
            cross_matches = [
                eid for eid in top_episode_ids
                if stored_eps_task.get(eid, current_task) != current_task
            ]
            cross_task_ratio = len(cross_matches) / len(top_episode_ids)

            # writer.log({f"Generalization/CrossTaskMatchRatio_t{timestep}": cross_task_ratio})
            print(f"🔎 Top {top_k} matched episodes for {episode_id} at t={timestep}: {top_episode_ids}")

            top_n = 10  # show top 10 most matched
            top_match_counts = sorted(self.match_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

            # match_freq_table = wandb.Table(columns=["Episode ID", "Count"])
            # frequency_dict = {}
            # for eid, count in top_match_counts:
            #     short_id = eid.split(":")[-1]  # keep only the last part of the elementId
            #     frequency_dict[short_id] = count
            # wandb.log({
            #         f"Match/FrequencyDict_t{timestep}": wandb.Table(
            #             data=[[k, v] for k, v in frequency_dict.items()],
            #             columns=["ID", "Count"]
            #         )
            # })

            # bar_chart = wandb.plot.bar(match_freq_table, "Episode ID", "Count",
            #                            title=f"Match Frequency (Top {top_n}) - t={timestep}")
            # writer.log({f"Match/FrequencyBar_Top{top_n}_t{timestep}": bar_chart})
        return G_t_set_encoding, top_episode_ids

    def process_state_node_data(self, node):
        node_type = set(node.labels).pop()
        node_id = node.element_id
        try:
            node_embedding = node._properties[self.embedding_mapping[node_type]]
        except:
            node_embedding = None
        return node_type, node_id, node_embedding

    def get_affordance_augmented_tensor(self, state_id, base_tensor):
        """Optional method to extract affordance-enhanced input from Neo4j and augment input tensor.
        """
        affs = self.wm.concept_space.get_affordances_for_state(state_id)
        aff_tensor = torch.tensor([aff['embedding'] for aff in affs], dtype=torch.float32)
        if aff_tensor.ndim == 1:
            aff_tensor = aff_tensor.unsqueeze(0)
        return torch.cat([base_tensor, aff_tensor.mean(dim=0, keepdim=True)], dim=2)

    def store_state_graph(self, encoded_state, state_id, episode_id, timestep, inventory, env):
        """
        Store current state graph into working memory (Neo4j) and also prepare partial SETLE graph trace.

        Arguments:
        - encoded_state: torch.Tensor for the current state
        - state_id: Neo4j node ID of the current state
        - episode_id: Neo4j ID of the episode node
        - timestep: current timestep
        - inventory: env.inventory at this timestep
        - env: full env instance (to read env_pos, goal, etc.)
        """
        # 1. Store meta information in Neo4j under the Episode node
        if timestep == 0:  # First step: create episode metadata
            self.wm.concept_space.set_property(episode_id, 'Episode', 'succesfull_outcome', False)
            self.wm.concept_space.set_property(episode_id, 'Episode', 'task', env.task_id, is_string=True)
            self.wm.concept_space.set_property(episode_id, 'Episode', 'inventory',
                                               ",".join(map(str, list(inventory))), is_string=True)
            self.wm.concept_space.set_property(episode_id, 'Episode', 'env_pos',
                                               json.dumps([list(s) for s in list(env.env_pos)]),
                                               is_string=True)

            dict_env = env.__dict__
            for key in dict_env.keys():
                try:
                    if key not in ['actions_taken', 'env_pos']:
                        self.wm.concept_space.set_property(episode_id, 'Episode', str(key), json.dumps(dict_env[key]))
                except Exception as e:
                    print(f"Could not store {key}: {e}")

        # 2. Generate MPS for this state
        mps = self.wm.generate_mps(state_id)

        # 3. Store state graph for partial SETLE matching
        self.episode_graph.append({'feats': encoded_state, 'mps': mps})


class TrainModel(object):

    def __init__(self, model, env, memory=(True, 100), writer=None,masked=True, params={}):
        self.model_to_train = model
        self.env = env
        self.use_memory = memory
        self.memory=None
        self.writer = writer
        self.params = params
        self.masked= masked

        self.object_extractor = SegmentAnythingObjectExtractor()
        self.action_embedder = ActionObservation()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.adapter = EnrichedStateAdapter().to(device)
        #self.concept_space = ConceptSpaceGDS(memory_type="afftest")
        self.wm = WorkingMemory(which_db='workingmemory')
        self.ltm = WorkingMemory(which_db="outcomesmall")
        self.use_actions_repr = True
        count_mps = 2

        args = heco_params()
        st_loader = StateLoader(nr_mps=2, mps=None)
        (batch_pos1, batch_pos2, batch_neg1), all_state_keys, all_aff_keys, all_obj_keys, action_keys, (
        fstate_p1, fstate_p2, fstate_n1) = st_loader.get_subgraph_episode_data(batch_size=1)
        feats = batch_pos1[0][0]
        nei_index = batch_pos1[0][1]
        mps = st_loader.generate_mps_episode(nei_index, fstate_p1)
        mps_dims = [mp.shape[1] for mp in mps]
        feats_dim_list = [i.shape[1] for i in batch_pos1[0][0]]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                     count_mps, args.sample_rate, args.nei_num, args.tau, args.lam, mps_dims).to(device)
        checkpoint = os.path.join(os.getcwd(),'ep_hybrid_1.5_14_0.14917626976966858.pkl')
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        self.setle = SetleEnricher(model, self.wm, self.ltm)

    def get_current_state_graph(self, observation, objects_interacting_frames,  episode_id, timestep):
        current_screen_objects, encoded_state, _ = self.object_extractor.extract_objects(observation)
        state_id, added_objs = self.wm.add_to_memory(encoded_state, current_screen_objects, episode_id, timestep)
        action_tool_ids, objs = self.wm.add_object_action_repr(objects_interacting_frames, state_id)
        return current_screen_objects, encoded_state, state_id, action_tool_ids


    def init_model(self, actions=0, checkpoint_file="", train_type='', count_iter=0):
        obs = self.env.reset()
        init_screen, state_enc, _ = self.object_extractor.extract_objects(obs)
        # init_screen = self.process_frames(obs)
        # _, _, screen_height, screen_width = init_screen.shape
        if actions == 0:
            n_actions = len(self.env.allowed_actions)
        else:
            n_actions = actions
        n_actions_cont = 2
        #objects_in_init_screen = self.object_extractor.extract_objects(init_screen.squeeze(0))
        policy_net = self.model_to_train(init_screen.shape, n_actions, n_actions_cont, self.env).to(device)
        target_net = self.model_to_train(init_screen.shape, n_actions, n_actions_cont, self.env).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.RMSprop(policy_net.parameters())
        optimizer = optim.Adam(policy_net.parameters(), lr=0.00001)


        if checkpoint_file != "":
            print(f"Trainning from checkpoint {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file)
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.use_memory[0] is not None:
            self.memory = ReplayMemory(self.use_memory[1])
            if train_type == 'baseline':
                self.train(target_net, policy_net, self.memory, self.params, optimizer, self.writer, count_iter=count_iter)
            else:
                self.train_setle(target_net, policy_net, self.memory, self.params, optimizer, self.writer, strategy_type=train_type, count_iter=count_iter)

        return

    # def compute_mask(self):
    #     sorter = np.argsort(self.env.allowed_actions)
    #     b = sorter[np.searchsorted(self.env.allowed_actions, self.env.inventory, sorter=sorter)]
    #     mask = np.zeros(self.env.allowed_actions.shape, dtype=bool)  # np.ones_like(a,dtype=bool)
    #     mask[b] = True
    #     return torch.tensor(mask, device=device), self.env.inventory

    # def compute_mask(self):
    #     allowed = self.env.allowed_actions  # global action list
    #     inventory = self.env.inventory  # current tools
    #
    #     # Build full-size binary mask (1 row, 939 columns)
    #     mask = torch.tensor([action in inventory for action in allowed], dtype=torch.bool, device=device)
    #     mask = mask.unsqueeze(0)  # shape becomes [1, 939] to match batch
    #     for i in range(mask.size(0)):
    #         if mask[i].sum() == 0:
    #             print(f"⚠️ WARNING: no valid actions for sample {i} — forcing default fallback.")
    #             mask[i, 0] = True
    #
    #     return mask, inventory

    def compute_mask(self):
        allowed = self.env.allowed_actions
        inventory = self.env.inventory

        # Create per-step mask
        row_mask = torch.tensor([action in inventory for action in allowed], dtype=torch.bool, device=device)  # [939]

        # Ensure at least one valid action
        if row_mask.sum() == 0:
            print("⚠️ No valid actions — using fallback index 0.")
            row_mask[0] = True

        return row_mask.unsqueeze(0), inventory  # [1, 939]

    # def select_action(self, state, params, policy_net, n_actions, steps_done):
    #     """
    #     Selects an action using ε-greedy strategy.
    #
    #     Args:
    #         state (torch.Tensor): Current state input
    #         params (dict): Contains epsilon parameters
    #         policy_net (nn.Module): The policy network
    #         n_actions (int): Number of discrete actions
    #         steps_done (int): Global steps counter
    #
    #     Returns:
    #         action (List): Selected action [idx, x, y]
    #         steps_done (int): Updated steps counter
    #         mask (Tensor): Action mask
    #         inventory (Tensor): Inventory tensor
    #     """
    #
    #     # Compute mask and inventory only once
    #     mask, inventory = self.compute_mask()
    #     inventory_tensor = torch.tensor(inventory, device=device).unsqueeze(0)
    #
    #     # Compute current epsilon using decay
    #     eps_threshold = params.get('eps_end', 0.05) + \
    #                     (params.get('eps_start', 0.9) - params.get('eps_end', 0.05)) * \
    #                     math.exp(-1. * steps_done / params.get('eps_decay', 200))
    #
    #     steps_done += 1
    #     sample = random.random()
    #
    #     if sample > eps_threshold:
    #         # Exploitation: choose best action from policy
    #         with torch.no_grad():
    #             q_vals_discrete, q_val_cont, action_sel = policy_net(state, mask, inventory_tensor.cpu())
    #             return action_sel, steps_done, mask, inventory_tensor
    #     else:
    #         # Exploration: random action (valid from inventory)
    #         random_action_idx = np.where(self.env.inventory == random.choice(self.env.inventory))[0][0]
    #         random_position = [random.uniform(-1, 1), random.uniform(-1, 1)]
    #         random_action = [[random_action_idx] + random_position]
    #         return random_action, steps_done, mask, inventory_tensor

    def select_action(self, state, params, policy_net, n_actions, steps_done):
        # Compute mask and inventory only once
        mask, inventory = self.compute_mask()
        inventory_tensor = torch.tensor(inventory, device=device).unsqueeze(0)

        eps_threshold = params.get('eps_end', 0.05) + \
                        (params.get('eps_start', 0.9) - params.get('eps_end', 0.05)) * \
                        math.exp(-1. * steps_done / params.get('eps_decay', 200))

        self.writer.log({"epsilon": eps_threshold})
        steps_done += 1
        sample = random.random()

        if sample > eps_threshold:
            with torch.no_grad():
                q_vals_discrete, q_val_cont, action_sel = policy_net(state, mask, inventory_tensor.cpu())

                if torch.isnan(q_vals_discrete).any():
                    print("⚠️ NaN detected in Q-values — falling back to random action")
                    return self._sample_random_action(inventory), steps_done, mask, inventory_tensor

                # Get index of best action from mask
                best_action_idx = q_vals_discrete.argmax(dim=1).item()

                # Map back to inventory index (in case allowed_actions is a subset)
                best_inventory_value = self.env.allowed_actions[best_action_idx]
                mapped_idx = np.where(self.env.inventory == best_inventory_value)[0][0]

                x, y = action_sel[0][1], action_sel[0][2]
                chosen_action = [[mapped_idx, x, y]]
                return chosen_action, steps_done, mask, inventory_tensor
        else:
            return self._sample_random_action(inventory), steps_done, mask, inventory_tensor

    def _sample_random_action(self, inventory):
        available_indices = np.arange(len(inventory))
        random_action_idx = int(np.random.choice(available_indices))
        random_position = [random.uniform(-1, 1), random.uniform(-1, 1)]
        return [[random_action_idx] + random_position]

    def compute_effect(self, st, st_plus_1, reward):
        difference = st_plus_1 - st
        combined_effect = torch.cat([difference.squeeze(0), torch.tensor(reward).unsqueeze(0).to(device)])
        return combined_effect

    def optimize_model(self, policy_net, target_net, params, memory, optimizer, loss_ep, writer):
        if len(memory) < params['batch_size']:
            return loss_ep

        transitions = memory.sample(params['batch_size'])
        batch = Transition(*zip(*transitions))

        # Prepare batches
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.tensor(batch.action).float().to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        reward_batch = torch.cat(batch.reward).float().to(device).clamp(-1.0, 1.0)
        mask_batch = torch.stack(batch.mask)
        inventory_batch = torch.stack(batch.inventory)

        print("🧪 mask_batch shape:", mask_batch.shape)
        print("🧪 Valid actions per row:", mask_batch.sum(dim=1))

        # Action breakdown
        num_categorical_actions = policy_net.num_discrete_actions
        # categorical_actions = action_batch[:, :num_categorical_actions]
        # continuous_actions = action_batch[:, num_categorical_actions:]

        categorical_actions = action_batch[:, 0].long().unsqueeze(1)  # shape: (B, 1)
        continuous_actions = action_batch[:, 1:]
        # shape: (B, 2)

        # Current Q-values
        q_values_categorical, q_values_continuous, _ = policy_net(state_batch, mask_batch, inventory_batch)

        # Extract Q-values for selected actions
        chosen_action_indices = categorical_actions.argmax(dim=1).long().unsqueeze(1)
        # chosen_q_values = q_values_categorical.gather(1, chosen_action_indices).squeeze(1)
        mapped_indices = []
        for i in range(categorical_actions.size(0)):
            inv_idx = categorical_actions[i].item()
            tool_id = inventory_batch[i][0][inv_idx].item()  # get tool ID
            if tool_id in self.env.allowed_actions:
                mapped_index = list(self.env.allowed_actions).index(tool_id)
            else:
                mapped_index = 0  # fallback if something weird happens
            mapped_indices.append(mapped_index)

        # Now gather using mapped indices
        mapped_tensor = torch.tensor(mapped_indices, device=device).unsqueeze(1)
        chosen_q_values = q_values_categorical.gather(1, mapped_tensor).squeeze(1)

        # chosen_q_values = q_values_categorical.gather(1, categorical_actions).squeeze(1)

        # chosen_q_values = q_values_categorical.gather(1, categorical_actions).squeeze(1)

        q_value_cont_selected = (q_values_continuous * continuous_actions).sum(dim=1)



        # Target Q-values (Double DQN compatible)
        with torch.no_grad():
            next_q_values_categorical, next_q_values_continuous, _ = target_net(next_state_batch, mask_batch,
                                                                                inventory_batch)
            max_q_vals, _ = next_q_values_categorical.max(dim=1)
            max_cont_q_vals = next_q_values_continuous.sum(dim=1)

        expected_state_action_values = reward_batch + params["gamma"] * max_q_vals
        expected_state_action_values_continuous = reward_batch + params["gamma"] * max_cont_q_vals

        # Loss computation
        categorical_loss = F.smooth_l1_loss(chosen_q_values, expected_state_action_values)
        continuous_loss = F.mse_loss(q_value_cont_selected, expected_state_action_values_continuous)

        total_loss = categorical_loss + continuous_loss

        # Log metrics
        writer.log({
            'Loss/categorical': categorical_loss.item(),
            'Loss/continuous': continuous_loss.item(),
            'Loss/total': total_loss.item(),
            'Q/mean_categorical': chosen_q_values.mean().item(),
            'Q/mean_continuous': q_value_cont_selected.mean().item(),
            'Reward/mean': reward_batch.mean().item()
        })

        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()

        return total_loss.item()

    def soft_update(self, target, source, tau=0.01):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)



    def log_trajectory_metrics(self, writer, step_count: int, total_reward: float,
                               redundant_actions: int, tool_switch_count: int,
                               used_tools, done: int, episode_index: int,
                               strategy_name: str = "default"):
        """
        Logs trajectory-level metrics to Weights & Biases.

        Parameters:
        - writer: WandB logger
        - step_count: Total steps taken in the episode
        - total_reward: Cumulative reward
        - redundant_actions: Count of repeated tools used in succession
        - tool_switch_count: Count of tool changes
        - used_tools: List of tool indices used in episode
        - done: Whether the episode was successful
        - episode_index: Current episode number
        - strategy_name: Optional label for the strategy used
        """
        unique_tools_used = len(set(used_tools))
        outcome = "Success" if done else "Failure"

        writer.log({
            f"{strategy_name}/Steps": step_count,
            f"{strategy_name}/Reward": total_reward,
            f"{strategy_name}/RedundantActions": redundant_actions,
            f"{strategy_name}/ToolSwitches": tool_switch_count,
            f"{strategy_name}/UniqueTools": unique_tools_used,
            f"{strategy_name}/Outcome": done,
            f"{strategy_name}/{outcome}Episodes": 1
        })

    def train(self, target_net, policy_net, memory, params, optimizer, writer, max_timesteps=14, count_iter=0):
        episode_durations = []
        num_episodes = 20
        steps_done = 0
        all_rewards = []
        all_lengths = []
        all_successes = []

        for i_episode in range(num_episodes):
            print(f"🟢 Episode {i_episode}")
            self.wm.concept_space.clear_wm()
            episode_id = self.wm.concept_space.add_data('Episode')['elementId(n)'][0]
            self.wm.concept_space.close()

            obs = self.env.reset()
            rew_ep = 0
            loss_ep = 0
            timestep = 0
            total_reward = 0
            step_count = 0

            redundant_actions = 0
            tool_switch_count = 0
            used_tools = []

            encoded_inventory, interacting_frames = self.action_embedder.get_inventory_embeddings(
                self.env.inventory, self.object_extractor
            )
            current_screen, encoded_state_t, state_id, action_tool_ids = self.get_current_state_graph(
                obs, interacting_frames, episode_id, timestep
            )

            for t in count():
                # Combine screen and inventory to form the input tensor
                input_tensor = torch.cat((current_screen, encoded_inventory.unsqueeze(0).unsqueeze(0)), dim=3)

                # Choose action with ε-greedy
                action, steps_done, mask, inventory_tensor = self.select_action(
                    input_tensor, params, policy_net,
                    len(self.env.allowed_actions), steps_done
                )

                # Apply action
                returned_state, reward, done, info = self.env.step(action[0])
                reward = torch.tensor([reward], device=device) if not isinstance(reward, torch.Tensor) else reward
                rew_ep += reward.item()

                step_count += 1
                total_reward += reward

                inventory_item = self.env.inventory.item(action[0][0])
                position = [action[0][1], action[0][2]]

                aff_id = self.wm.match_action_affordance(
                    action_tool_ids, inventory_item, position, reward.item(), timestep
                )
                self.wm.concept_space.match_state_add_aff(state_id, aff_id)

                writer.log({"Action taken": action[0][0]})

                # Observe next state
                next_screen, encoded_state_t_plus_1, state_id, action_tool_ids = self.get_current_state_graph(
                    returned_state, interacting_frames, episode_id, timestep + 1
                )
                # Track redundant actions
                if len(used_tools) > 0 and used_tools[-1] == action[0][0]:
                    redundant_actions += 1

                # Track tool switch
                if len(used_tools) > 0 and used_tools[-1] != action[0][0]:
                    tool_switch_count += 1

                # Append current tool
                used_tools.append(action[0][0])

                affordance_effect = self.compute_effect(encoded_state_t, encoded_state_t_plus_1, reward.item())

                if not done:
                    next_state = next_screen
                else:
                    next_state = None

                if next_state is not None:
                    mask = mask.squeeze(0)
                    memory.push(current_screen, action[0], next_state, reward, mask, inventory_tensor)

                current_screen = next_state
                encoded_state_t = encoded_state_t_plus_1

                # Optimization step
                loss_ep = loss_ep+self.optimize_model(policy_net, target_net, params, memory, optimizer, loss_ep, writer)

                if aff_id and state_id:
                    self.wm.concept_space.match_state_add_aff_outcome(state_id, aff_id)
                    self.wm.concept_space.set_property(
                        aff_id, 'Affordance', 'outcome', affordance_effect.tolist()
                    )

                timestep += 1

                if done or t == max_timesteps:
                    episode_durations.append(t + 1)
                    writer.log({
                        "Reward episode": rew_ep,
                        "Episode duration": t + 1,
                        "Train loss": loss_ep / (t + 1),
                        "Episode index": i_episode
                    })
                    self.log_trajectory_metrics(writer,t,rew_ep,redundant_actions, tool_switch_count, used_tools, info['cur_goal_hit'], i_episode, 'baseline')
                    all_rewards.append(total_reward)
                    all_lengths.append(step_count)
                    all_successes.append(info['cur_goal_hit'])
                    break

                if t % 100 == 0:
                    torch.save({
                        'episode': i_episode,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f"checkpoints/baseline{count_iter}_ep_{i_episode}_{t}.pt")


            # Periodic target network update
            if i_episode % params['target_update'] == 0:
                target_net.load_state_dict(policy_net.state_dict())
            torch.cuda.empty_cache()
            self.log_final_statistics(writer, all_rewards, all_lengths, all_successes, 'baseline', count_iter)

        return

    # def train_setle(self, target_net, policy_net, memory, params, optimizer, writer, max_timesteps=14):
    #     episode_durations = []
    #     num_episodes = 3000
    #     steps_done = 0
    #     counter = 0
    #     smallest_loss = 99999
    #     loss = 0
    #     for i_episode in range(num_episodes):
    #         print(f"Episode:{i_episode}")
    #         # clear wm before new Episode
    #         self.wm.concept_space.clear_wm()
    #         concept_space_episode_id = self.wm.concept_space.add_data('Episode')
    #         episode_id = concept_space_episode_id['elementId(n)'][0]
    #         self.wm.concept_space.close()
    #         episode_memory = []
    #         # Initialize the environment and state
    #         obs = self.env.reset()
    #         rew_ep = 0
    #         loss_ep = 0
    #         losses = []
    #         timestep = 0
    #         aff_id = None
    #         state_id = None
    #         encoded_inventory, objects_interacting_frames = self.action_embedder.get_inventory_embeddings(
    #             self.env.inventory, self.object_extractor)
    #         current_screen, encoded_state_t, state_id, action_tool_ids = self.get_current_state_graph(
    #             obs, objects_interacting_frames,
    #             episode_id, timestep
    #         )
    #
    #         for t in count():
    #             # # 1. Store the graph representation for SETLE trace building
    #             # self.setle.store_state_graph(encoded_state_t, state_id, episode_id, t, self.env.inventory, self.env)  # <-- NEW
    #
    #             # 3. Match partial trace to LTM and enrich graph at state_id
    #             if self.params.get("use_setle_graphs", True):  # <-- NEW
    #                 if t >= 3:
    #                     matched_episodes = self.setle.match_episodes(episode_id, self.writer, timestep=t, top_k=3)  # <-- NEW
    #                     matched_episodes_full = []
    #                     for matched_id in matched_episodes:
    #                         ep_matched = self.ltm.concept_space.get_episode_graph(matched_id)
    #                         matched_episodes_full.append(ep_matched)
    #                     print(matched_episodes)
    #
    #             output_tensor = torch.cat((current_screen, encoded_inventory.unsqueeze(0).unsqueeze(0)), dim=3)
    #             action, steps_done, mask, inventory = self.select_action(output_tensor, params, policy_net,
    #                                                                      len(self.env.allowed_actions), steps_done)
    #
    #             returned_state, reward, done, _ = self.env.step(action[0])
    #             inventory_item_applied = self.env.inventory.item(action[0][0])
    #             position_applied = [action[0][1], action[0][2]]
    #             try:
    #                 r = reward.item()
    #             except:
    #                 r = reward
    #                 print(reward)
    #
    #             aff_id = self.wm.match_action_affordance(action_tool_ids, inventory_item_applied, position_applied, r,
    #                                                      timestep)
    #             self.wm.concept_space.match_state_add_aff(state_id, aff_id)
    #
    #             self.writer.log({"Action taken": action[0][0]})
    #             reward = torch.tensor([r], device=device)
    #
    #             rew_ep += r
    #             # current_screen, encoded_state = self.object_extractor.extract_objects(returned_state)
    #             current_screen, encoded_state_t_plus_1, state_id, action_tool_ids = self.get_current_state_graph(
    #                 returned_state, objects_interacting_frames,
    #                 episode_id, timestep + 1
    #             )
    #             affordance_effect = self.compute_effect(encoded_state_t, encoded_state_t_plus_1, r)
    #             episode_memory.append(current_screen)
    #
    #             if not done:
    #                 next_state = current_screen
    #             else:
    #                 next_state = None
    #
    #             # Store the transition in memory
    #             # self.wm.compute_attention(timestep, episode_id)
    #             if next_state is not None:
    #                 memory.push(current_screen, action[0], next_state, reward, mask, inventory)
    #
    #             # Move to the next state
    #             state = next_state
    #
    #             # Perform one step of the optimization (on the target network)
    #             loss_ep = self.optimize_model(policy_net, target_net, params, memory, optimizer, loss_ep, writer)
    #
    #             if done or t == max_timesteps:
    #                 episode_durations.append(t + 1)
    #                 self.writer.log(
    #                     {"Reward episode": rew_ep, "Episode duration": t + 1, "Train loss": loss_ep / (t + 1)})
    #                 # print(loss_ep / (t + 1))
    #                 # episode_frames_wandb = make_grid(episode_frames)
    #                 # images = wandb.Image(episode_frames_wandb, caption=f'Episode {i_episode} states')
    #                 # self.writer.log({'states': episode_frames})
    #
    #                 break
    #             timestep += 1
    #             # state_id = self.wm.add_to_memory(encoded_state, current_screen, episode_id, timestep)
    #             # action_tool_ids = self.wm.add_object_action_repr(objects_interacting_frames, state_id)
    #
    #             if aff_id is not None and state_id is not None:
    #                 self.wm.concept_space.match_state_add_aff_outcome(state_id, aff_id)
    #                 self.wm.concept_space.set_property(aff_id, 'Affordance', 'outcome', affordance_effect.tolist())
    #             # Update the target network, copying all weights and biases in DQN
    #         if i_episode % params['target_update'] == 0:
    #             target_net.load_state_dict(policy_net.state_dict())
    #         # if i_episode % 150 == 0 and i_episode != 0:
    #         #     self.evaluate(target_net, writer, i_episode)
    #         # if i_episode % 100 == 0 and i_episode != 0:
    #         #     PATH = f"model_{i_episode}_{loss_ep}.pt"
    #         #     torch.save({
    #         #         'epoch': i_episode,
    #         #         'model_state_dict': policy_net.state_dict(),
    #         #         'optimizer_state_dict': optimizer.state_dict(),
    #         #         'loss': loss_ep,
    #         #     }, PATH)
    #     return



    def analyze_trajectory(self, trajectory, rewards, goal_reached):
        """
        Analyze trajectory efficiency and log relevant metrics.

        Args:
            trajectory (list): List of (tool_id, x, y) tuples.
            rewards (list): List of rewards per step.
            goal_reached (bool): Whether the goal was achieved.

        Returns:
            dict: A dictionary of trajectory metrics.
        """
        from collections import Counter
        trajectory_length = len(trajectory)
        total_reward = sum(rewards)

        # Tool usage
        tools_used = [tool for tool, _, _ in trajectory]
        tool_switches = sum(1 for i in range(1, len(tools_used)) if tools_used[i] != tools_used[i - 1])
        tool_counts = Counter(tools_used)

        # Redundant = using same tool with no reward effect
        redundant_actions = 0
        for i in range(1, len(tools_used)):
            if tools_used[i] == tools_used[i - 1] and rewards[i] == 0:
                redundant_actions += 1

        return {
            "Trajectory/length": trajectory_length,
            "Trajectory/goal_reached": int(goal_reached),
            "Trajectory/total_reward": total_reward,
            "Trajectory/tool_switches": tool_switches,
            "Trajectory/redundant_actions": redundant_actions,
            "Trajectory/unique_tools": len(tool_counts),
        }

    def train_setle(self, target_net, policy_net, memory, params, optimizer, writer, max_timesteps=14, strategy_type='soft_update', count_iter=0):
        episode_durations = []
        num_episodes = 40
        steps_done = 0

        all_rewards = []
        all_lengths = []
        all_successes = []

        for i_episode in range(num_episodes):
            print(f"🚀 Episode {i_episode}")
            self.wm.concept_space.clear_wm()
            concept_space_episode_id = self.wm.concept_space.add_data('Episode')
            episode_id = concept_space_episode_id['elementId(n)'][0]
            self.wm.concept_space.close()

            obs = self.env.reset()
            timestep = 0
            loss_ep = 0
            rew_ep = 0
            done = False
            aff_id = None
            step_count = 0
            redundant_actions = 0
            tool_switch_count = 0
            used_tools = []
            total_reward = 0

            encoded_inventory, objects_interacting_frames = self.action_embedder.get_inventory_embeddings(
                self.env.inventory, self.object_extractor
            )
            current_screen, encoded_state_t, state_id, action_tool_ids = self.get_current_state_graph(
                obs, objects_interacting_frames, episode_id, timestep
            )

            for t in count():
                # 🧠 Match & enrich from LTM
                if self.params.get("use_setle_graphs", True) and t >= 3:
                    if strategy_type in ['adapter_and_penalty','adpter_penalty_soft_update']:
                        gt_encoding, matched_eps = self.setle.match_episodes(episode_id, writer, timestep=t, top_k=3, use_penalty=True)
                    else:
                        gt_encoding, matched_eps = self.setle.match_episodes(episode_id, writer, timestep=t, top_k=3, use_penalty=False)


                    if gt_encoding is None:
                        break
                    all_matched_graphs = []
                    for matched_id in matched_eps:
                        ep_nodes, ep_rel = self.ltm.concept_space.get_episode_graph(matched_id)
                        all_matched_graphs.extend(ep_nodes)

                    node_type_groups = {"Affordance": [], "ObjectConcept": [], "ActionRepr": []}
                    for node in all_matched_graphs:
                        node_type = set(node.labels).pop()
                        if node_type in node_type_groups and (hasattr(node,"_properties") and "value" in node._properties or hasattr(node,"_properties") and "outcome" in node._properties ):
                            node_id = node.element_id
                            emb = node._properties["value"] if 'value' in node._properties else node._properties['outcome']
                            node_type_groups[node_type].append((node_id, torch.tensor(emb, dtype=torch.float32),node))

                    for node_type, node_data in node_type_groups.items():
                        if node_data:
                            candidates, full_nodes = zip(*[(a[:2], a[2]) for a in node_data])
                            self.setle.apply_attention_enrichment(
                                query_embedding=gt_encoding,
                                candidates=candidates,
                                full_nodes=full_nodes,
                                attn_module=self.setle.attn_modules[node_type],
                                state_id=state_id,
                                node_type=node_type,
                                writer=writer,
                                step=t
                            )

                        # 🧠 Re-encode after enrichment
                    enriched_encoding = self.setle.ltm_init.encode_episode(episode_id)

                # 🔁 Re-encode enriched current state graph using SETLE
                enriched_encoding_state = self.setle.ltm_init.encode_single_state_ep(state_id)  # shape: [1, 64]

                if enriched_encoding_state is None:
                    print(f"⚠️ Could not encode state {state_id}, skipping action selection...")
                    continue  # or fallback to default action selection
                if strategy_type in ['adapter_and_penalty','adpter_penalty_soft_update']:
                    adapted_encoding_state = self.adapter(enriched_encoding_state)
                    state_input = adapted_encoding_state
                else:
                    state_input = enriched_encoding_state
                # 🎯 Select action using enriched representation



                action, steps_done, mask, inventory = self.select_action(
                    state_input,  # shape [1, 1, 1, 64] to mimic [B, C, H, W]
                    params,
                    policy_net,
                    len(self.env.allowed_actions),
                    steps_done
                )


                # # 🎯 Select action
                # output_tensor = torch.cat((current_screen, encoded_inventory.unsqueeze(0).unsqueeze(0)), dim=3)
                # action, steps_done, mask, inventory = self.select_action(
                #     output_tensor, params, policy_net, len(self.env.allowed_actions), steps_done
                # )

                obs, reward, done, info = self.env.step(action[0])
                r = reward.item() if isinstance(reward, torch.Tensor) else reward
                reward_tensor = torch.tensor([r], device=device)

                step_count += 1
                total_reward += r

                # Track redundant actions
                if len(used_tools) > 0 and used_tools[-1] == action[0][0]:
                    redundant_actions += 1

                # Track tool switch
                if len(used_tools) > 0 and used_tools[-1] != action[0][0]:
                    tool_switch_count += 1

                # Append current tool
                used_tools.append(action[0][0])

                inventory_item_applied = self.env.inventory.item(action[0][0])
                position_applied = [action[0][1], action[0][2]]

                aff_id = self.wm.match_action_affordance(
                    action_tool_ids, inventory_item_applied, position_applied, r, timestep
                )
                self.wm.concept_space.match_state_add_aff(state_id, aff_id)
                self.writer.log({"Action taken": action[0][0]})
                rew_ep += r

                next_screen, encoded_state_t_plus_1, state_id, action_tool_ids = self.get_current_state_graph(
                    obs, objects_interacting_frames, episode_id, timestep + 1
                )
                affordance_effect = self.compute_effect(encoded_state_t, encoded_state_t_plus_1, r)
                encoded_state_t = encoded_state_t_plus_1

                enriched_encoding_state_t_plus_1 = self.setle.ltm_init.encode_single_state_ep(state_id)  # shape: [1, 64]
                adapted_encoding_state_t_plus_1 = self.adapter(enriched_encoding_state_t_plus_1)



                # 💾 Store transition
                if not done:
                    if strategy_type == "action_selection_only":
                        memory.push(current_screen.detach(), action[0], next_screen.detach(), reward_tensor, mask, inventory)
                        loss_ep += self.optimize_model(policy_net, target_net, params, memory, optimizer, loss_ep, writer)

                    else:
                        memory.push(state_input.detach(), action[0], adapted_encoding_state_t_plus_1.detach(), reward_tensor, mask, inventory)
                        loss_ep += self.optimize_model(policy_net, target_net, params, memory, optimizer, loss_ep, writer)

                timestep += 1

                if aff_id is not None and state_id is not None:
                    self.wm.concept_space.match_state_add_aff_outcome(state_id, aff_id)
                    self.wm.concept_space.set_property(aff_id, 'Affordance', 'outcome', affordance_effect.tolist())

                if t % params['target_update'] == 0:
                    if strategy_type in ['adpter_penalty_soft_update']:
                        self.soft_update(target_net, policy_net, tau=0.01)
                    else:
                        target_net.load_state_dict(policy_net.state_dict())

                if t % 100 == 0:
                    torch.save({
                        'episode': i_episode,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f"checkpoints/{strategy_type}{count_iter}_ep_{i_episode}_{t}.pt")

                if done or t == max_timesteps:
                    writer.log({
                        "Reward episode": rew_ep,
                        "Episode duration": t + 1,
                        "Train loss": loss_ep / (t + 1)
                    })

                    unique_tools_used = len(set(used_tools))

                    # writer.log({
                    #     "Trajectory/Steps": step_count,
                    #     "Trajectory/Reward": total_reward,
                    #     "Trajectory/RedundantActions": redundant_actions,
                    #     "Trajectory/ToolSwitches": tool_switch_count,
                    #     "Trajectory/UniqueTools": unique_tools_used,
                    # })
                    self.log_trajectory_metrics(writer,t,rew_ep,redundant_actions, tool_switch_count, used_tools, info['cur_goal_hit'], i_episode, strategy_type)
                    all_rewards.append(total_reward)
                    all_lengths.append(step_count)
                    all_successes.append(info['cur_goal_hit'])

                    break
            torch.cuda.empty_cache()

            self.log_final_statistics(writer, all_rewards, all_lengths, all_successes, strategy_type, count_iter)


            # # 🔄 If new episode has novel embedding → add to LTM
            # if self.setle.should_store_episode(episode_id):
            #     self.ltm.concept_space.copy_episode_from(self.wm.concept_space, episode_id)\

    def log_final_statistics(self, writer, all_rewards, all_lengths, all_successes, strategy_name="default", count_iter=0):
        """
        Logs final summary statistics across all episodes of a strategy.

        Parameters:
        - writer: wandb logger
        - all_rewards: list of total rewards per episode
        - all_lengths: list of trajectory lengths (steps) per episode
        - all_successes: list of 1/0 indicating if each episode was a success
        - strategy_name: name of the strategy (used for logging key prefixes)
        """
        num_episodes = len(all_rewards)
        avg_reward = sum(all_rewards) / num_episodes if num_episodes > 0 else 0
        avg_length = sum(all_lengths) / num_episodes if num_episodes > 0 else 0
        success_rate = sum(all_successes) / num_episodes if num_episodes > 0 else 0

        writer.log({
            f"FinalStats_{strategy_name}_{count_iter}/AverageReward": avg_reward,
            f"FinalStats_{strategy_name}_{count_iter}/AverageLength": avg_length,
            f"FinalStats_{strategy_name}_{count_iter}/SuccessRate": success_rate,
            f"FinalStats_{strategy_name}_{count_iter}/TotalEpisodes": num_episodes,
            f"FinalStats_{strategy_name}_{count_iter}/TotalSuccesses": sum(all_successes),
            f"FinalStats_{strategy_name}_{count_iter}/TotalFailures": num_episodes - sum(all_successes),
        })






def main():
    args = sys.argv[1:]
    checkpoint_file = ""
    if len(args) >0 and args[0] == '-checkpoint':
        checkpoint_file = args[1]
    params = {
        'batch_size': 10,
        'gamma': 0.99,
         'eps_start': 1.0,
    'eps_end': 0.1,
    'eps_decay': 1000,   #
        'target_update': 30
    }

    import os, tempfile




    # env = gym.make(f'CreateLevelPush-v0')
    # settings = CreateGameSettings(
    #     evaluation_mode=True,
    #     max_num_steps=30,
    #     action_set_size=7,
    #     render_mega_res=False,
    #     render_ball_traces=False)
    # env.set_settings(settings)
    # env.reset()
    done = False
    frames = []
    strategy_types = ['baseline','action_selection_only','setle_action_selection_optimisation','adapter_and_penalty','adpter_penalty_soft_update']
    tasks = ["CreateLevelPush", "CreateLevelBelt",  "CreateLevelObstacle", "CreateLevelBuckets", "CreateLevelBasket",  "CreateLevelObstacle"]
    strategy = strategy_types[4]
    nr_runs = 5
    for run in range(nr_runs):
        task = tasks[run]
        env = gym.make(f'{task}-v0')
        settings = CreateGameSettings(
            evaluation_mode=True,
            max_num_steps=30,
            action_set_size=7,
            render_mega_res=False,
            render_ball_traces=False)
        env.set_settings(settings)
        env.reset()
        print(f'Strategy {strategy} and task {task}')
        wandb_media_dir = os.path.join(tempfile.gettempdir(), "wandb-media")
        os.makedirs(wandb_media_dir, exist_ok=True)
        wandb_logger = Logger(f"{strategy}_run_{run}_task_{task}",
                          project='setle_stats')
        logger = wandb_logger.get_logger()
        if strategy in ['baseline', 'action_selection_only']:
            trainer = TrainModel(DQN,
                                 env, (True, 1000),
                                 logger, True, params)
            trainer.init_model(checkpoint_file=checkpoint_file, train_type=strategy, count_iter=run)
            wandb.finish()
            # try:
            #     trainer = TrainModel(DQN,
            #                      env, (True, 1000),
            #                      logger, True, params)
            #     trainer.init_model(checkpoint_file=checkpoint_file, train_type=strategy, count_iter=run)
            # except Exception as e:
            #     torch.cuda.empty_cache()
            #     print(e)
            #     continue

        else:
            trainer = TrainModel(DQNSetle,
                                 env, (True, 1000),
                                 logger, True, params)
            trainer.init_model(checkpoint_file=checkpoint_file, train_type=strategy, count_iter=run)
            wandb.finish()
            # try:
            #     trainer = TrainModel(DQNSetle,
            #                      env, (True, 1000),
            #                      logger, True, params)
            #     trainer.init_model(checkpoint_file=checkpoint_file, train_type=strategy, count_iter=run)
            # except Exception as e:
            #     torch.cuda.empty_cache()
            #     print(e)
            #     continue








    return

if __name__ == '__main__':
    main()



