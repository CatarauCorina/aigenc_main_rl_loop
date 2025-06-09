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
from co_segment_anything.dqn_sam_minigrid import DQN, DQNSetle
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

# os.environ['NEO4J_BOLT_URL']='bolt://localhost:7687'
# os.environ['NEO_PASS']='minigrid123'
# os.environ['NEO_USER']='neo4j'
MINIGRID_ACTIONS = [
    "left",     # 0
    "right",    # 1
    "forward",  # 2
    "pickup",   # 3
    "drop",     # 4
    "toggle",   # 5
    "done"      # 6
]

tasks = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-UnlockPickup-v0",
    "MiniGrid-MultiRoom-N4-S5-v0",
    "MiniGrid-SimpleCrossingS9N1-v0"
]


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




from collections import namedtuple

TransitionMinigrid = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemoryMinigrid:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        """Save a MiniGrid transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = TransitionMinigrid(state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_td(self, batch_size):
        """Returns batches for TD loss computation."""
        transitions = self.sample(batch_size)
        batch = TransitionMinigrid(*zip(*transitions))

        # next_state is None for final steps
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=device, dtype=torch.long)  # MiniGrid: int actions
        reward_batch = torch.cat(batch.reward)

        return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask

    def __len__(self):
        return len(self.memory)





class SetleEnricher:
    def __init__(self, encoder, wm, ltm):
        self.embedding_mapping = {
            'ObjectConcept': 'value',
            'Affordance': 'outcome',
            'StateT': 'state_enc'
        }
        self.encoder = encoder  # SETLE encoder (e.g., HeCo)
        self.wm = wm            # WorkingMemory instance
        self.ltm_init = LTMInitliser(use_memory='workingmemory', has_action_repr=False, tasks=tasks, minigrid_mem='ltm2')
        self.match_tracking_table = wandb.Table(columns=["Timestep", "ShortID", "Frequency"])

        self.ltm = ltm
        self.match_counts = defaultdict(int)  # track how often each episode is matched

        self.attn_modules = {
            "Affordance": EnrichmentAttention(query_dim=64, candidate_dim=513).to(device),
            "ObjectConcept": EnrichmentAttention(query_dim=64, candidate_dim=512).to(device),
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


    def match_episodes(self, episode_id, writer, top_k=5, timestep=0, use_penalty=True):
        # Step 1: extract recent subgraph from the partial trace and encode it
        G_t_set_encoding = self.ltm_init.encode_episode_minigird(episode_id)

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


from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
class TrainModel(object):

    def __init__(self, model, env, memory=(True, 100), writer=None,masked=True, params={}):
        self.model_to_train = model
        self.env = env
        self.use_memory = memory
        self.memory=None
        self.writer = writer
        self.masked= masked
        self.params = params


        self.object_extractor = SegmentAnythingObjectExtractor()

        # checkpoint_path = "../co_segment_anything/checkpoints"
        #
        # sam = sam_model_registry["vit_b"](checkpoint=os.path.join(checkpoint_path, "sam_vit_b_01ec64.pth"))
        # from PIL import Image
        # import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



        self.adapter = EnrichedStateAdapter().to(device)
        #self.concept_space = ConceptSpaceGDS(memory_type="afftest")
        self.wm = WorkingMemory(which_db='workingmemory')
        self.ltm = WorkingMemory(which_db="ltm2")
        self.use_actions_repr = False
        count_mps = 2

        args = heco_params()
        st_loader = StateLoader(nr_mps=2, mps=None, use_memory='ltm2', tasks=tasks)
        (batch_pos1, batch_pos2, batch_neg1), all_state_keys, all_aff_keys, all_obj_keys, (
        fstate_p1, fstate_p2, fstate_n1) = st_loader.get_subgraph_episode_data_minigrid(batch_size=1)
        feats = batch_pos1[0][0]
        nei_index = batch_pos1[0][1]
        mps = st_loader.generate_mps_episode(nei_index, fstate_p1, has_action_repr=False)
        mps_dims = [mp.shape[1] for mp in mps]
        feats_dim_list = [i.shape[1] for i in batch_pos1[0][0]]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                     count_mps, args.sample_rate, args.nei_num, args.tau, args.lam, mps_dims,minigrid=True).to(device)
        checkpoint = os.path.join(os.getcwd(),'ep_hybrid_1.5_29_0.6739.pkl')
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        self.setle = SetleEnricher(model, self.wm, self.ltm)


    # def get_current_state_graph(
    #         self,
    #         observation,
    #         episode_id,
    #         timestep,
    #         prev_state_id=None,
    #         prev_aff_node_id=None,
    #         prev_encoded_state=None,
    #         prev_action_name=None,
    #         reward=0.0
    # ):
    #     """
    #     MiniGrid RL: Adds State node, optional outcome edge, Affordance node + influences edge.
    #     """
    #     # 1️⃣ Extract objects + features for current state
    #     current_screen_objects, encoded_state, _ = self.object_extractor.extract_objects(observation)
    #     state_id, added_objs = self.wm.add_to_memory(encoded_state, current_screen_objects, episode_id, timestep)
    #
    #     # 2️⃣ If previous step exists, add Affordance → State outcome edge
    #     if prev_state_id is not None and prev_aff_node_id is not None:
    #         effect = self.compute_effect(prev_encoded_state, encoded_state, reward)
    #         self.wm.concept_space.minigrid_add_edge(
    #             prev_aff_node_id,
    #             state_id,
    #             label="outcome",
    #             source_tag="minigrid",
    #             properties={"effect": effect.tolist()}
    #         )
    #
    #     # 3️⃣ Add current Affordance node + influences edge from State
    #     aff_label = f"{prev_action_name}" if prev_action_name else "unknown_action"
    #     aff_node_id = self.wm.concept_space.minigrid_create_node(
    #         "Affordance",
    #         {
    #             "label": aff_label,
    #             "action": prev_action_name,
    #             "reward": reward
    #         }
    #     )
    #     self.wm.concept_space.minigrid_add_edge(
    #         state_id,
    #         aff_node_id,
    #         label="influences",
    #         source_tag="minigrid"
    #     )
    #
    #     return current_screen_objects, encoded_state, state_id, aff_node_id

    def get_current_state_graph(
            self,
            observation,
            episode_id,
            timestep,
            prev_state_id=None,
            prev_aff_node_id=None,
            prev_encoded_state=None,
            prev_action_name=None,
            reward=0.0,
            all_obj=[]
    ):
        """
        MiniGrid RL: Adds State node, optional outcome edge, Affordance node + influences edge.
        """
        # ✅ 1️⃣ Add state + objects via existing function
        all_obj, encoded_state, state_id, all_obj = self.add_state_node_only(
            observation,
            episode_id,
            timestep,
            all_obj=all_obj
        )

        aff_node_id = None

        # ✅ 2️⃣ If action was taken (prev_state exists), create affordance + connect edges
        if prev_state_id is not None and prev_action_name is not None:
            aff_label = prev_action_name

            # Create affordance node
            effect = self.compute_effect(prev_encoded_state, encoded_state, reward)

            aff_node_id = self.wm.concept_space.minigrid_create_node(
                "Affordance",
                {
                    "label": aff_label,
                    "action": prev_action_name,
                    "reward": reward,
                    'outcome': effect.tolist(),

                }
            )

            # Connect previous state to affordance
            self.wm.concept_space.minigrid_add_edge(
                prev_state_id,
                aff_node_id,
                label="influences",
                source_tag="minigrid"
            )

            # Connect affordance to current state (outcome)
            effect = self.compute_effect(prev_encoded_state, encoded_state, reward)
            self.wm.concept_space.minigrid_add_edge(
                aff_node_id,
                state_id,
                label="outcome",
                source_tag="minigrid",
            )

        return all_obj, encoded_state, state_id, aff_node_id

    def is_black_square(self, obj_crop, black_thresh=100, squareness_tol=0.2):
        """
        Check if an object crop is a black square.

        Parameters:
            obj_crop (np.ndarray): Cropped image of the object.
            black_thresh (int): Max pixel value to consider as "black" (0–255).
            squareness_tol (float): Allowed relative tolerance for width ≈ height.

        Returns:
            bool: True if the object is a black square, False otherwise.
        """
        if obj_crop.size == 0:
            return False

        # Convert to grayscale if it's RGB
        if obj_crop.ndim == 3:
            gray = np.mean(obj_crop, axis=2)
        else:
            gray = obj_crop

        # Check how many pixels are under the black threshold
        black_pixels_ratio = np.mean(gray < black_thresh)

        # Check if the shape is roughly square
        h, w = gray.shape
        squareness = abs(h - w) / max(h, w)

        return black_pixels_ratio > 0.9 and squareness < squareness_tol

    def clip_extraction(self, pil_img):
        # Labels
        labels = ["a red key", "a red triangle", "wall", "a black and yellow square with line", "empty black square",
                  'a green square', 'gray wall', 'a golden key']
        try:
            inputs = self.clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            predicted = labels[logits_per_image.argmax()]
        except Exception as e:
            return 'unknown'
        return predicted

    def extract_and_embed_objects(self, obs, show=True, use_labels=False):
        import cv2

        original_h, original_w = obs.shape[:2]
        obs_resized = cv2.resize(obs, (96, 96))

        scale_x = 96 / original_w
        scale_y = 96 / original_h

        masks = self.object_extractor.mask_generator.generate(np.array(obs_resized))
        object_embeddings = []
        i = 0
        objects_in_image = []
        object_labels = []
        for m in masks:  # Limit to top 3 masks

            x, y, w, h = m['bbox']
            if m['area'] < 20:
                continue
            if m['stability_score'] < 0.9:
                continue
            if m['predicted_iou'] < 0.85:
                continue
            # ✅ Rescale bbox to match resized image
            x0 = int(x * scale_x)
            y0 = int(y * scale_y)
            x1 = int((x + w) * scale_x)
            y1 = int((y + h) * scale_y)

            # x, y, w, h = m['bbox']
            # x0, y0 = int(x), int(y)
            # x1, y1 = int(x + w), int(y + h)

            # Validate that the bbox is valid (i.e., width and height are positive)
            if x1 > x0 and y1 > y0:
                obj_crop = obs[y0:y1, x0:x1]
            else:
                print(f"Skipping invalid bbox: {m['bbox']}")

            # obj_crop = obs[int(y0):int(y1), int(x0):int(x1)]

            if obj_crop.size == 0:
                continue
            if obj_crop.size == 0:
                continue

                # Display the object crop
            if not self.is_black_square(obj_crop):
                if show:
                    plt.imshow(obj_crop)
                    plt.title(f"Object {i + 1} - Crop")
                    plt.axis('off')
                    plt.show()
                # img_tensor = transform(Image.fromarray(obj_crop)).unsqueeze(0).to(device)
                # img_tensor = resnet_transform(obj_crop).to(device)
                pil_image = Image.fromarray(obj_crop.astype(np.uint8))
                # if use_labels:
                #     label = self.clip_extraction(pil_image)
                #
                #     img_tensor = self.object_extractor.resnet_transform(pil_image).to(device)
                #     i = i + 1
                #     if label != 'unknown':
                #         objects_in_image.append(img_tensor)
                #         object_labels.append(label)
                # else:
                img_tensor = self.object_extractor.resnet_transform(pil_image).to(device)
                objects_in_image.append(img_tensor)

        with torch.no_grad():
            tensor_all_objs = torch.stack(objects_in_image).to(device)
            emb = self.object_extractor.pass_through_resnet(tensor_all_objs)

        return emb, object_labels

    def extract_and_embed_objects_test(self, obs, show=False, use_labels=False):
        import cv2
        from PIL import Image
        import matplotlib.pyplot as plt
        count_walls = 0
        # ✅ Step 1: Enhance contrast
        obs = cv2.resize(obs, (96, 96))  # upscale for better SAM performance
        lab = cv2.cvtColor(obs, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        obs = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # ✅ Step 2: Generate masks with SAM
        masks = self.object_extractor.mask_generator.generate(obs)

        object_embeddings = []
        objects_in_image = []
        object_labels = []

        for i, m in enumerate(masks):
            # ✅ Step 3: Apply strict filters
            if m['area'] < 30:
                continue
            if m['stability_score'] < 0.9:
                continue
            if m['predicted_iou'] < 0.85:
                continue



            x, y, w, h = m['bbox']
            x0, y0 = int(x), int(y)
            x1, y1 = int(x + w), int(y + h)

            if x1 <= x0 or y1 <= y0:
                continue

            obj_crop = obs[y0:y1, x0:x1]
            if obj_crop.size == 0:
                continue
            if self.is_black_square(obj_crop):
                continue

            if np.std(obj_crop) == 0 and count_walls > 0:
                continue
            elif np.std(obj_crop) == 0:
                count_walls+=1

            # ✅ Step 4: Visualize (optional)
            if show:
                plt.imshow(obj_crop)
                plt.title(f"Object {i + 1} - Crop")
                plt.axis('off')
                plt.show()

            # ✅ Step 5: Prepare for embedding
            pil_image = Image.fromarray(obj_crop.astype(np.uint8))
            img_tensor = self.object_extractor.resnet_transform(pil_image).to(device)

            # ✅ Step 6: Optional CLIP label
            # if use_labels:
            #     label = self.clip_extraction(pil_image)
            #     if label != 'unknown':
            #         objects_in_image.append(img_tensor)
            #         object_labels.append(label)
            # else:
            objects_in_image.append(img_tensor)

        # ✅ Step 7: Return embeddings
        if not objects_in_image:
            return torch.empty(0), object_labels

        with torch.no_grad():
            tensor_all_objs = torch.stack(objects_in_image).to(device)
            emb = self.object_extractor.pass_through_resnet(tensor_all_objs)

        return emb, object_labels

    def add_state_node_only(self, observation, episode_id, timestep, all_obj=[]):
        tensor_img_reduced = self.object_extractor.resnet_transform_frame(observation).float().to(device)
        encoded_state = self.object_extractor.pass_through_resnet(tensor_img_reduced.unsqueeze(0))
        state_id = self.wm.concept_space.minigrid_add_state(episode_id, timestep=timestep, embedding=encoded_state.squeeze(0).tolist())


        if len(all_obj) == 0:
            print("Creating")
            emb_list, label_list = self.extract_and_embed_objects_test(observation)
            emb_list = emb_list.tolist()
            i = 0
            for emb in emb_list:
                if len(label_list) > 0:
                    label = label_list[i]
                else:
                    label = None
                similar_obj_ids, all_obj = self.wm.concept_space.find_similar_object_concepts(emb, use_label=False, check_label=label,
                                                                       fetch_obj=all_obj)
                if similar_obj_ids:
                    for sid in similar_obj_ids:
                        self.wm.concept_space.add_edge(
                            source=state_id,
                            target=sid,
                            label="has_object",
                            weight=1.0,
                            state_id=state_id,
                        )
                else:
                    id_obj = self.wm.concept_space.minigrid_add_object_concept(state_id, emb, label)
                    all_obj.append((id_obj, emb))
        else:
            print('Just connecting')
            for obj_id, _ in all_obj:
                self.wm.concept_space.add_edge(
                    source=state_id,
                    target=obj_id,
                    label="has_object",
                    weight=1.0,
                    state_id=state_id
                )


        return all_obj, encoded_state, state_id, all_obj

    def init_model(self, actions=0, checkpoint_file="", train_type='', count_iter=0):
        obs = self.env.reset()
        init_screen = torch.tensor(self.env.render()).permute(2, 0, 1)
        n_actions = len(MINIGRID_ACTIONS)

        #objects_in_init_screen = self.object_extractor.extract_objects(init_screen.squeeze(0))
        policy_net = self.model_to_train(init_screen.shape, n_actions).to(device)
        target_net = self.model_to_train(init_screen.shape, n_actions).to(device)
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
            self.memory = ReplayMemoryMinigrid(self.use_memory[1])
            if train_type == 'baseline':
                self.train(target_net, policy_net, self.memory, self.params, optimizer, self.writer, count_iter=count_iter)
            else:
                self.train_setle(target_net, policy_net, self.memory, self.params, optimizer, self.writer, strategy_type=train_type, count_iter=count_iter)

        return



    def select_action(self, state, params, policy_net, n_actions, steps_done):
        """
        SETLE RL action selection adapted for MiniGrid.
        Only discrete actions, no inventory or continuous coordinates.
        """
        eps_threshold = params.get('eps_end', 0.05) + \
                        (params.get('eps_start', 0.9) - params.get('eps_end', 0.05)) * \
                        math.exp(-1. * steps_done / params.get('eps_decay', 200))

        self.writer.log({"epsilon": eps_threshold})
        steps_done += 1
        sample = random.random()

        if sample > eps_threshold:
            with torch.no_grad():
                # In MiniGrid: expect policy_net to return only discrete q-values
                q_vals_discrete = policy_net(state)  # assume output shape [1, n_actions]

                if torch.isnan(q_vals_discrete).any():
                    print("⚠️ NaN detected in Q-values — falling back to random action")
                    return self._sample_random_action(n_actions), steps_done

                action = q_vals_discrete.argmax(dim=1).item()
                return action, steps_done
        else:
            return self._sample_random_action(n_actions), steps_done

    def _sample_random_action(self, n_actions):
        return random.randint(0, n_actions - 1)

    def compute_effect(self, st, st_plus_1, reward):
        difference = st_plus_1 - st
        combined_effect = torch.cat([difference.squeeze(0), torch.tensor(reward).unsqueeze(0).to(device)])
        return combined_effect

    def optimize_model(self, policy_net, target_net, params, memory, optimizer, loss_ep, writer):
        if len(memory) < params['batch_size']:
            return loss_ep

        transitions = memory.sample(params['batch_size'])
        batch = TransitionMinigrid(*zip(*transitions))

        # Prepare batches
        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.tensor(batch.action).long().to(device)  # action is int
        next_state_batch = torch.stack(batch.next_state).to(device)
        reward_batch = torch.stack(batch.reward).float().to(device).clamp(-1.0, 1.0)

        # Forward pass for current states
        q_values = policy_net(state_batch)  # shape: [B, n_actions]
        chosen_q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Forward pass for next states (Double DQN compatible)
        with torch.no_grad():
            next_q_values = target_net(next_state_batch)
            max_next_q_values, _ = next_q_values.max(dim=1)
            expected_q_values = reward_batch.squeeze(1) + params["gamma"] * max_next_q_values

        # Loss computation
        loss = F.smooth_l1_loss(chosen_q_values, expected_q_values)

        # Log metrics
        writer.log({
            'Loss/Q-Loss': loss.item(),
            'Q/mean': chosen_q_values.mean().item(),
            'Reward/mean': reward_batch.mean().item()
        })

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()

        return loss.item()

    def soft_update(self, target, source, tau=0.01):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def log_trajectory_metrics(self, writer, step_count: int, total_reward: float,
                               redundant_actions: int, done: int, episode_index: int,
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
        outcome = "Success" if done else "Failure"

        writer.log({
            f"{strategy_name}/Steps": step_count,
            f"{strategy_name}/Reward": total_reward,
            f"{strategy_name}/RedundantActions": redundant_actions,
            f"{strategy_name}/Outcome": done,
            f"{strategy_name}/{outcome}Episodes": 1
        })

    def train(self, target_net, policy_net, memory, params, optimizer, writer, max_timesteps=9, count_iter=0):
        episode_durations = []
        num_episodes = 250
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
            obs_data, _ = obs
            obs_img = self.env.render()
            all_obj = []

            # Initialize graph for first state
            current_objs, encoded_state, state_id, all_obj = self.add_state_node_only(
                observation=obs_img,
                episode_id=episode_id,
                timestep=0,
                all_obj=all_obj
            )
            # Initialize tracking
            last_state_id = state_id
            last_encoded_state = encoded_state
            last_aff_node_id = None
            last_action_name = None

            total_reward = 0
            step_count = 0

            for t in count():
                action, steps_done = self.select_action(
                    torch.tensor(obs_img, dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0), params, policy_net,
                    n_actions=self.env.action_space.n,
                    steps_done=steps_done
                )
                action_name = MINIGRID_ACTIONS[action]

                # TAKE ACTION
                next_obs, reward, done, _, _ = self.env.step(action)
                # next_obs_img = next_obs['image']
                next_obs_img = self.env.render()
                reward_tensor = torch.tensor([reward], device=device)
                rew_ep += reward

                step_count += 1
                total_reward += reward

                # ADD GRAPH NODE FOR NEXT STATE (+ edges from last affordance)
                current_objs, next_encoded_state, next_state_id, next_aff_node_id = self.get_current_state_graph(
                    observation=next_obs_img,
                    episode_id=episode_id,
                    timestep=t + 1,
                    prev_state_id=last_state_id,
                    prev_aff_node_id=last_aff_node_id,
                    prev_encoded_state=last_encoded_state,
                    prev_action_name=action_name,
                    reward=reward,
                    all_obj=all_obj
                )

                # ADD RL TRANSITION TO MEMORY


                memory.push(
                    torch.tensor(obs_img, dtype=torch.float32).permute(2, 0, 1).to(device),  # current state
                    action,
                    torch.tensor(next_obs_img, dtype=torch.float32).permute(2, 0, 1).to(device),  # next state
                    reward_tensor
                )

                # RL OPTIMIZATION STEP
                loss_ep += self.optimize_model(policy_net, target_net, params, memory, optimizer, loss_ep, writer)

                # PREPARE for next step
                last_state_id = next_state_id
                last_encoded_state = next_encoded_state
                last_aff_node_id = next_aff_node_id
                last_action_name = action_name

                rew_ep += reward
                obs_img = next_obs_img

                if done or t == max_timesteps:
                    episode_durations.append(t + 1)
                    writer.log({
                        "Reward episode": rew_ep,
                        "Episode duration": t + 1,
                        "Train loss": loss_ep / (t + 1),
                        "Episode index": i_episode
                    })
                    torch.cuda.empty_cache()

                    break

                # Periodic target update
            if i_episode % params['target_update'] == 0:
                target_net.load_state_dict(policy_net.state_dict())

        return

    def train_setle(self, target_net, policy_net, memory, params, optimizer, writer, max_timesteps=14,
                    strategy_type='soft_update', count_iter=0):
        num_episodes = 250
        steps_done = 0
        all_rewards = []
        all_lengths = []
        all_successes = []

        for i_episode in range(num_episodes):
            print(f"🚀 Episode {i_episode}")
            self.wm.concept_space.clear_wm()
            episode_id = self.wm.concept_space.add_data('Episode')['elementId(n)'][0]
            self.wm.concept_space.close()

            obs = self.env.reset()
            obs_img = self.env.render()
            all_obj = []

            # ✅ Initial state graph
            all_obj, encoded_state, state_id, all_obj = self.add_state_node_only(
                observation=obs_img,
                episode_id=episode_id,
                timestep=0,
                all_obj=all_obj
            )
            last_state_id = state_id
            last_encoded_state = encoded_state
            last_aff_node_id = None
            last_action_name = None

            total_reward = 0
            rew_ep = 0
            step_count = 0
            loss_ep = 0


            for t in range(max_timesteps):
                # ✅ SETLE enrichment (only difference vs train)
                if self.params.get("use_setle_graphs", True) and t >= 3:
                    if strategy_type in ['adapter_and_penalty', 'adpter_penalty_soft_update']:
                        gt_encoding, matched_eps = self.setle.match_episodes(
                            episode_id, writer, timestep=t, top_k=3, use_penalty=True
                        )
                    else:
                        gt_encoding, matched_eps = self.setle.match_episodes(
                            episode_id, writer, timestep=t, top_k=3, use_penalty=False
                        )

                    if gt_encoding is not None:
                        all_matched_graphs = []
                        for matched_id in matched_eps:
                            ep_nodes, _ = self.ltm.concept_space.get_episode_graph_minigrid(matched_id)
                            all_matched_graphs.extend(ep_nodes)

                        node_type_groups = {"Affordance": [], "ObjectConcept": []}
                        for node in all_matched_graphs:
                            node_type = set(node.labels).pop()
                            if node_type in node_type_groups and hasattr(node, "_properties"):
                                emb = node._properties.get("value") or node._properties.get("outcome")
                                if emb is not None:
                                    node_type_groups[node_type].append(
                                        (node.element_id, torch.tensor(emb, dtype=torch.float32), node)
                                    )

                        for node_type, node_data in node_type_groups.items():
                            if node_data:
                                candidates, full_nodes = zip(*[(a[:2], a[2]) for a in node_data])
                                self.setle.apply_attention_enrichment(
                                    query_embedding=gt_encoding,
                                    candidates=candidates,
                                    full_nodes=full_nodes,
                                    attn_module=self.setle.attn_modules[node_type],
                                    state_id=last_state_id,
                                    node_type=node_type,
                                    writer=writer,
                                    step=t
                                )

                # 🎯 Action selection
                enriched_encoding = self.setle.ltm_init.encode_single_state_ep_minigrid(last_state_id)
                if enriched_encoding is None:
                    continue

                state_input = (
                    self.adapter(enriched_encoding)
                    if strategy_type in ['adapter_and_penalty', 'adpter_penalty_soft_update']
                    else enriched_encoding
                )

                action, steps_done = self.select_action(
                    state_input, params, policy_net,
                    self.env.action_space.n, steps_done
                )
                action_name = MINIGRID_ACTIONS[action]

                # 🎮 Step environment
                next_obs, reward, done,_, _ = self.env.step(action)
                next_obs_img = self.env.render()
                reward_tensor = torch.tensor([reward], device=device)

                rew_ep += reward

                step_count += 1
                r = reward.item() if isinstance(reward, torch.Tensor) else reward

                total_reward += r

                # ✅ Graph update via standard get_current_state_graph()
                all_obj, next_encoded_state, next_state_id, aff_node_id = self.get_current_state_graph(
                    observation=next_obs_img,
                    episode_id=episode_id,
                    timestep=t + 1,
                    prev_state_id=last_state_id,
                    prev_aff_node_id=last_aff_node_id,
                    prev_encoded_state=last_encoded_state,
                    prev_action_name=action_name,
                    reward=r,
                    all_obj=all_obj
                )

                # 💾 Store transition
                # if not done:
                next_enriched = self.setle.ltm_init.encode_single_state_ep_minigrid(next_state_id)
                next_input = (
                        self.adapter(next_enriched)
                        if strategy_type != "action_selection_only"
                        else next_enriched
                    )
                memory.push(state_input.squeeze(1).detach(), action, next_input.squeeze(1).detach(), reward_tensor)
                loss_ep += self.optimize_model(policy_net, target_net, params, memory, optimizer, loss_ep, writer)

                # 📝 Prep next step
                last_state_id = next_state_id
                last_encoded_state = next_encoded_state
                last_aff_node_id = aff_node_id
                last_action_name = action_name
                rew_ep += r

                # 🔁 Target network update
                if t % params['target_update'] == 0:
                    if strategy_type == 'adpter_penalty_soft_update':
                        self.soft_update(target_net, policy_net, tau=0.01)
                    else:
                        target_net.load_state_dict(policy_net.state_dict())

                # 💾 Checkpoint
                if t % 100 == 0:
                    torch.save({
                        'episode': i_episode,
                        'model_state_dict': policy_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f"checkpoints/{strategy_type}{count_iter}_ep_{i_episode}_{t}.pt")

                if done:
                    break

            # ✅ End of episode logging
            success = 1 if done and rew_ep > 0 else 0
            with self.wm.concept_space.driver.session() as session:
                session.run(
                    "MATCH (e:Episode) WHERE elementId(e) = $eid SET e.successful_outcome = $success_flag",
                    eid=episode_id,
                    success_flag=success
                )

            writer.log({
                "Reward episode": rew_ep,
                "Episode duration": step_count,
                "Train loss": loss_ep / step_count if step_count > 0 else 0
            })
            all_rewards.append(total_reward)
            all_lengths.append(step_count)
            all_successes.append(success)
            torch.cuda.empty_cache()

        # ✅ Final summary
        self.log_final_statistics(writer, all_rewards, all_lengths, all_successes, strategy_type, count_iter)

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

    # def train_setle(self, target_net, policy_net, memory, params, optimizer, writer, max_timesteps=14,
    #                 strategy_type='soft_update', count_iter=0):
    #     num_episodes = 40
    #     steps_done = 0
    #     all_rewards = []
    #     all_lengths = []
    #     all_successes = []
    #
    #     for i_episode in range(num_episodes):
    #         print(f"🚀 Episode {i_episode}")
    #         self.wm.concept_space.clear_wm()
    #         episode_id = self.wm.concept_space.add_data('Episode')['elementId(n)'][0]
    #         self.wm.concept_space.close()
    #
    #         obs = self.env.reset()
    #         obs_visual= self.env.render()
    #         timestep = 0
    #         loss_ep = 0
    #         rew_ep = 0
    #         done = False
    #         step_count = 0
    #         total_reward = 0
    #         all_obj = []
    #
    #         # ✅ Step 0: Add initial state only
    #         all_obj, encoded_state, state_id, all_obj = self.add_state_node_only(
    #             obs, episode_id, timestep=0, all_obj=all_obj
    #         )
    #         last_state_id = state_id
    #         last_encoded_state = encoded_state
    #         last_action_name = None
    #
    #         for t in range(max_timesteps):
    #             # 🧠 SETLE Enrichment
    #             if self.params.get("use_setle_graphs", True) and t >= 3:
    #                 if strategy_type in ['adapter_and_penalty', 'adpter_penalty_soft_update']:
    #                     gt_encoding, matched_eps = self.setle.match_episodes(episode_id, writer, timestep=t, top_k=3,
    #                                                                          use_penalty=True)
    #                 else:
    #                     gt_encoding, matched_eps = self.setle.match_episodes(episode_id, writer, timestep=t, top_k=3,
    #                                                                          use_penalty=False)
    #
    #                 if gt_encoding is not None:
    #                     all_matched_graphs = []
    #                     for matched_id in matched_eps:
    #                         ep_nodes, _ = self.ltm.concept_space.get_episode_graph(matched_id)
    #                         all_matched_graphs.extend(ep_nodes)
    #
    #                     node_type_groups = {"Affordance": [], "ObjectConcept": []}
    #                     for node in all_matched_graphs:
    #                         node_type = set(node.labels).pop()
    #                         if node_type in node_type_groups and hasattr(node, "_properties"):
    #                             emb = node._properties.get("value") or node._properties.get("outcome")
    #                             if emb is not None:
    #                                 node_type_groups[node_type].append(
    #                                     (node.element_id, torch.tensor(emb, dtype=torch.float32), node))
    #
    #                     for node_type, node_data in node_type_groups.items():
    #                         if node_data:
    #                             candidates, full_nodes = zip(*[(a[:2], a[2]) for a in node_data])
    #                             self.setle.apply_attention_enrichment(
    #                                 query_embedding=gt_encoding,
    #                                 candidates=candidates,
    #                                 full_nodes=full_nodes,
    #                                 attn_module=self.setle.attn_modules[node_type],
    #                                 state_id=last_state_id,
    #                                 node_type=node_type,
    #                                 writer=writer,
    #                                 step=t
    #                             )
    #
    #             # 🔁 Re-encode enriched state
    #             enriched_encoding = self.setle.ltm_init.encode_single_state_ep(last_state_id)
    #             if enriched_encoding is None:
    #                 continue
    #
    #             state_input = self.adapter(enriched_encoding) if strategy_type in ['adapter_and_penalty',
    #                                                                                'adpter_penalty_soft_update'] else enriched_encoding
    #
    #             # 🎯 Select action
    #             action, steps_done = self.select_action(
    #                 state_input, params, policy_net,
    #                 self.env.action_space.n, steps_done
    #             )
    #             action_name = MINIGRID_ACTIONS[action]
    #
    #             # 🎮 Apply action
    #             next_obs, reward, done, info = self.env.step(action)
    #             r = reward.item() if isinstance(reward, torch.Tensor) else reward
    #             reward_tensor = torch.tensor([r], device=device)
    #
    #             step_count += 1
    #             total_reward += r
    #
    #             # ✅ ADD CORRECT GRAPH CONNECTIONS
    #             # → 1. Affordance node
    #             aff_node_id = self.wm.concept_space.minigrid_create_node(
    #                 "Affordance",
    #                 {"label": action_name, "action": action_name, "reward": r}
    #             )
    #             # → 2. last_state → influences → Affordance
    #             self.wm.concept_space.minigrid_add_edge(
    #                 last_state_id, aff_node_id, label="influences", source_tag="minigrid"
    #             )
    #             # → 3. Add new state
    #             all_obj, next_encoded_state, next_state_id, all_obj = self.add_state_node_only(
    #                 next_obs, episode_id, timestep=t + 1, all_obj=all_obj
    #             )
    #             # → 4. Affordance → outcome → new state
    #             effect = self.compute_effect(last_encoded_state, next_encoded_state, r)
    #             self.wm.concept_space.minigrid_add_edge(
    #                 aff_node_id, next_state_id, label="outcome", source_tag="minigrid",
    #                 properties={"effect": effect.tolist()}
    #             )
    #
    #             # 💾 Store transition
    #             if not done:
    #                 next_enriched = self.setle.ltm_init.encode_single_state_ep(next_state_id)
    #                 next_input = self.adapter(
    #                     next_enriched) if strategy_type != "action_selection_only" else next_enriched
    #                 memory.push(state_input.detach(), action, next_input.detach(), reward_tensor)
    #                 loss_ep += self.optimize_model(policy_net, target_net, params, memory, optimizer, loss_ep, writer)
    #
    #             # 📝 Prep next step
    #             last_state_id = next_state_id
    #             last_encoded_state = next_encoded_state
    #             last_action_name = action_name
    #             rew_ep += r
    #
    #             # 🔁 Target update
    #             if t % params['target_update'] == 0:
    #                 if strategy_type == 'adpter_penalty_soft_update':
    #                     self.soft_update(target_net, policy_net, tau=0.01)
    #                 else:
    #                     target_net.load_state_dict(policy_net.state_dict())
    #
    #             # 💾 Checkpoint
    #             if t % 100 == 0:
    #                 torch.save({
    #                     'episode': i_episode,
    #                     'model_state_dict': policy_net.state_dict(),
    #                     'optimizer_state_dict': optimizer.state_dict(),
    #                 }, f"checkpoints/{strategy_type}{count_iter}_ep_{i_episode}_{t}.pt")
    #
    #             if done:
    #                 break
    #
    #         # ✅ End of episode: success label
    #         success = 1 if done and rew_ep > 0 else 0
    #         with self.wm.concept_space.driver.session() as session:
    #             session.run(
    #                 "MATCH (e:Episode) WHERE elementId(e) = $eid SET e.successful_outcome = $success_flag",
    #                 eid=episode_id,
    #                 success_flag=success
    #             )
    #
    #         writer.log({
    #             "Reward episode": rew_ep,
    #             "Episode duration": step_count,
    #             "Train loss": loss_ep / step_count if step_count > 0 else 0
    #         })
    #         all_rewards.append(total_reward)
    #         all_lengths.append(step_count)
    #         all_successes.append(success)
    #         torch.cuda.empty_cache()
    #
    #     self.log_final_statistics(writer, all_rewards, all_lengths, all_successes, strategy_type, count_iter)

    # def train_setle(self, target_net, policy_net, memory, params, optimizer, writer, max_timesteps=14,
    #                 strategy_type='soft_update', count_iter=0):
    #     num_episodes = 40
    #     steps_done = 0
    #     all_rewards = []
    #     all_lengths = []
    #     all_successes = []
    #
    #     for i_episode in range(num_episodes):
    #         print(f"🚀 Episode {i_episode}")
    #         self.wm.concept_space.clear_wm()
    #         episode_id = self.wm.concept_space.add_data('Episode')['elementId(n)'][0]
    #         self.wm.concept_space.close()
    #
    #         obs = self.env.reset()
    #         timestep = 0
    #         loss_ep = 0
    #         rew_ep = 0
    #         done = False
    #         step_count = 0
    #         total_reward = 0
    #
    #         # ✅ Add initial state only
    #         current_objs, encoded_state, state_id = self.add_state_node_only(obs, episode_id, timestep=0)
    #         last_state_id = state_id
    #         last_encoded_state = encoded_state
    #         last_aff_node_id = None
    #         last_action_name = None
    #
    #         for t in range(max_timesteps):
    #             # 🧠 SETLE Enrichment
    #             if self.params.get("use_setle_graphs", True) and t >= 3:
    #                 if strategy_type in ['adapter_and_penalty', 'adpter_penalty_soft_update']:
    #                     gt_encoding, matched_eps = self.setle.match_episodes(episode_id, writer, timestep=t, top_k=3,
    #                                                                          use_penalty=True)
    #                 else:
    #                     gt_encoding, matched_eps = self.setle.match_episodes(episode_id, writer, timestep=t, top_k=3,
    #                                                                          use_penalty=False)
    #
    #                 if gt_encoding is not None:
    #                     all_matched_graphs = []
    #                     for matched_id in matched_eps:
    #                         ep_nodes, ep_rel = self.ltm.concept_space.get_episode_graph(matched_id)
    #                         all_matched_graphs.extend(ep_nodes)
    #
    #                     node_type_groups = {"Affordance": [], "ObjectConcept": []}
    #                     for node in all_matched_graphs:
    #                         node_type = set(node.labels).pop()
    #                         if node_type in node_type_groups and hasattr(node, "_properties"):
    #                             emb = node._properties.get("value") or node._properties.get("outcome")
    #                             if emb is not None:
    #                                 node_type_groups[node_type].append(
    #                                     (node.element_id, torch.tensor(emb, dtype=torch.float32), node))
    #
    #                     for node_type, node_data in node_type_groups.items():
    #                         if node_data:
    #                             candidates, full_nodes = zip(*[(a[:2], a[2]) for a in node_data])
    #                             self.setle.apply_attention_enrichment(
    #                                 query_embedding=gt_encoding,
    #                                 candidates=candidates,
    #                                 full_nodes=full_nodes,
    #                                 attn_module=self.setle.attn_modules[node_type],
    #                                 state_id=last_state_id,
    #                                 node_type=node_type,
    #                                 writer=writer,
    #                                 step=t
    #                             )
    #
    #             # 🔁 Re-encode state after enrichment
    #             enriched_encoding = self.setle.ltm_init.encode_single_state_ep(last_state_id)
    #             if enriched_encoding is None:
    #                 continue
    #
    #             # Optionally apply adapter
    #             state_input = self.adapter(enriched_encoding) if strategy_type in ['adapter_and_penalty',
    #                                                                                'adpter_penalty_soft_update'] else enriched_encoding
    #
    #             # 🎯 Select action
    #             action, steps_done = self.select_action(
    #                 state_input,
    #                 params,
    #                 policy_net,
    #                 self.env.action_space.n,
    #                 steps_done
    #             )
    #             action_name = MINIGRID_ACTIONS[action]
    #
    #             # 🎮 Apply action
    #             next_obs, reward, done, info = self.env.step(action)
    #             r = reward.item() if isinstance(reward, torch.Tensor) else reward
    #             reward_tensor = torch.tensor([r], device=device)
    #
    #             step_count += 1
    #             total_reward += r
    #
    #             # ✅ Add graph elements
    #             aff_node_id = self.wm.concept_space.minigrid_create_node("Affordance", {"label": action_name, "action": action_name, "reward": r})
    #             self.wm.concept_space.minigrid_add_edge(last_state_id, aff_node_id, label="influences", source_tag="minigrid")
    #
    #             # Add next state
    #             current_objs, next_encoded_state, next_state_id = self.add_state_node_only(next_obs, episode_id,
    #                                                                                        timestep=t + 1)
    #
    #             # Add outcome edge
    #             effect = self.compute_effect(last_encoded_state, next_encoded_state, r)
    #             self.wm.concept_space.minigrid_add_edge(aff_node_id, next_state_id, label="outcome", source_tag="minigrid",
    #                         properties={"effect": effect.tolist()})
    #
    #             # 💾 Store transition
    #             if not done:
    #                 next_enriched = self.setle.ltm_init.encode_single_state_ep(next_state_id)
    #                 next_input = self.adapter(next_enriched) if strategy_type != "action_selection_only" else None
    #                 memory.push(state_input.detach(), action,
    #                             next_input.detach() if next_input is not None else next_enriched.detach(),
    #                             reward_tensor)
    #                 loss_ep += self.optimize_model(policy_net, target_net, params, memory, optimizer, loss_ep, writer)
    #
    #             # 📝 Prep next step
    #             last_state_id = next_state_id
    #             last_encoded_state = next_encoded_state
    #             last_aff_node_id = aff_node_id
    #             last_action_name = action_name
    #             rew_ep += r
    #
    #             # 🔁 Target network update
    #             if t % params['target_update'] == 0:
    #                 if strategy_type == 'adpter_penalty_soft_update':
    #                     self.soft_update(target_net, policy_net, tau=0.01)
    #                 else:
    #                     target_net.load_state_dict(policy_net.state_dict())
    #
    #             # 💾 Save checkpoint
    #             if t % 100 == 0:
    #                 torch.save({
    #                     'episode': i_episode,
    #                     'model_state_dict': policy_net.state_dict(),
    #                     'optimizer_state_dict': optimizer.state_dict(),
    #                 }, f"checkpoints/{strategy_type}{count_iter}_ep_{i_episode}_{t}.pt")
    #
    #             if done:
    #                 break
    #
    #         # ✅ End of episode: write outcome
    #         if done and rew_ep > 0:
    #             success= 1
    #         elif done:
    #             success = 0
    #         with self.wm.concept_space.driver.session() as session:
    #             session.run(
    #                 "MATCH (e:Episode) WHERE elementId(e) = $eid SET e.successful_outcome = $success_flag",
    #                 eid=episode_id,
    #                 success_flag=success
    #             )
    #
    #         writer.log({
    #             "Reward episode": rew_ep,
    #             "Episode duration": step_count,
    #             "Train loss": loss_ep / step_count if step_count > 0 else 0
    #         })
    #         all_rewards.append(total_reward)
    #         all_lengths.append(step_count)
    #         all_successes.append(success)
    #         torch.cuda.empty_cache()
    #
    #     self.log_final_statistics(writer, all_rewards, all_lengths, all_successes, strategy_type, count_iter)

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




from gym_minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper


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



    done = False
    frames = []
    strategy_types = ['baseline','action_selection_only','setle_action_selection_optimisation','adapter_and_penalty','adpter_penalty_soft_update']
    # "MiniGrid-Empty-5x5-v0",
    # "MiniGrid-DoorKey-5x5-v0",
    # "MiniGrid-UnlockPickup-v0",

    tasks = [
          "MiniGrid-Empty-5x5-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-UnlockPickup-v0",
        "MiniGrid-SimpleCrossingS9N1-v0"
    ]
    strategy = strategy_types[4]
    nr_runs = 4
    for run in range(nr_runs):
        torch.cuda.empty_cache()
        task = tasks[run]
        env = gym.make(task, render_mode="rgb_array")  # MiniGrid environments
        env = FullyObsWrapper(env)  # recommended for full observable state

        print(f'Strategy {strategy} and task {task}')

        wandb_media_dir = os.path.join(tempfile.gettempdir(), "wandb-media")
        os.makedirs(wandb_media_dir, exist_ok=True)

        wandb_logger = Logger(f"{strategy}_run_{run}_task_{task}", project='setle_rl_minigrid_stats')
        logger = wandb_logger.get_logger()

        # ✅ Initialize SETLE RL trainer for MiniGrid
        if strategy in ['baseline', 'action_selection_only']:
            trainer = TrainModel(DQN,
                                 env, (True, 1000),
                                 logger, True, params)

            trainer.init_model(checkpoint_file=checkpoint_file, train_type=strategy, count_iter=run)
            wandb.finish()


        elif strategy in ['setle_action_selection_optimisation','adapter_and_penalty', 'adpter_penalty_soft_update']:
            trainer = TrainModel(DQNSetle,
                                 env, (True, 1000),
                                 logger, True, params)
            trainer.init_model(checkpoint_file=checkpoint_file, train_type=strategy, count_iter=run)
            wandb.finish()




    return

if __name__ == '__main__':
    main()



