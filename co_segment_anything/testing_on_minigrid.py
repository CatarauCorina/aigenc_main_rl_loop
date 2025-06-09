import gym
import numpy as np
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
from minigrid_oracle import  SmarterOracle
from gym_minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper

import json
from datetime import datetime as dt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torchvision import transforms
from torchvision.models import vgg16
from neo4j import GraphDatabase
# from sentence_transformers import SentenceTransformer

# Setup SAM + VGG16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "../co_segment_anything/checkpoints"

sam = sam_model_registry["vit_b"](checkpoint=os.path.join(checkpoint_path,"sam_vit_b_01ec64.pth"))
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(model=sam)

vgg_model = vgg16(pretrained=True).features.to(device)
vgg_model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
from sam_utils import SegmentAnythingObjectExtractor
from memory_graph.gds_concept_space import ConceptSpaceGDS

segmentor = SegmentAnythingObjectExtractor()
cs = ConceptSpaceGDS(memory_type='ltm3')

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
OBJECT_MAP = {
    'wall': 'wall',
    'door': 'door',
    'key': 'key',
    'ball': 'ball',
    'box': 'box',
    'goal': 'goal',
    None: 'empty'
}

MINIGRID_ACTIONS = [
    "left",     # 0
    "right",    # 1
    "forward",  # 2
    "pickup",   # 3
    "drop",     # 4
    "toggle",   # 5
    "done"      # 6
]

# Setup Neo4j
class GraphStore:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def add_episode(self, success, task):
        with self.driver.session() as session:
            result = session.run(
                "USE ltm3 CREATE (e:Episode {succesfull_outcome: $success, task: $task}) RETURN elementId(e) AS eid",
                success=success, task=task
            )
            return result.single()["eid"]

    def add_state(self, episode_id, timestep, embedding):
        with self.driver.session() as session:
            result = session.run(
                """
                USE ltm3
                MATCH (e:Episode) WHERE elementId(e) = $eid
                CREATE (s:StateT)
                CREATE (e)-[:has_state {t: $timestep, state_enc: $emb}]->(s)
                RETURN elementId(s) AS sid
                """, eid=episode_id, timestep=timestep, emb=embedding
            )
            return result.single()["sid"]

    def add_object_concept(self, state_id, emb, label):
        with self.driver.session() as session:
            result = session.run(
                """
                USE ltm3
                MATCH (s:StateT) WHERE elementId(s) = $sid
                CREATE (o:ObjectConcept {value: $embedding, name: $label})
                CREATE (s)-[:has_object]->(o)
                RETURN elementId(o) AS oid
                """, sid=state_id, embedding=emb, label=label
            )
            return result.single()["oid"]

    def add_edge(self, source, target, label, weight=1.0, state_id=None, source_tag=None):
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
                USE ltm3
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

    def create_node(self, label, props, source="manual"):
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
            USE ltm3
            CREATE (n:{label})
            SET n += $props, n.source = $source
            RETURN elementId(n) AS node_id
            """
            result = session.run(query, props=props, source=source)
            return result.single()["node_id"]


import numpy as np


def is_black_square(obj_crop, black_thresh=30, squareness_tol=0.2):
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


# SAM extraction + VGG embedding

def clip_extraction(pil_img):
    # Labels
    labels = ["a red key", "a red triangle", "wall", "a black and yellow square with line", "empty black square",'a green square', 'gray wall','a golden key']
    try:
        inputs = processor(text=labels, images=pil_img, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        predicted = labels[logits_per_image.argmax()]
    except Exception as e:
        return 'unknown'
    return predicted


def extract_and_embed_objects(obs, show=False, use_labels=False):
    masks = mask_generator.generate(np.array(obs))
    object_embeddings = []
    i = 0
    objects_in_image = []
    object_labels = []
    for m in masks:  # Limit to top 3 masks
        x, y, w, h = m['bbox']
        x0, y0 = int(x), int(y)
        x1, y1 = int(x + w), int(y + h)

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
        if not is_black_square(obj_crop):
            if show:
                plt.imshow(obj_crop)
                plt.title(f"Object {i + 1} - Crop")
                plt.axis('off')
                plt.show()
            # img_tensor = transform(Image.fromarray(obj_crop)).unsqueeze(0).to(device)
            # img_tensor = resnet_transform(obj_crop).to(device)
            pil_image = Image.fromarray(obj_crop.astype(np.uint8))
            if use_labels:
                label = clip_extraction(pil_image)

                img_tensor = segmentor.resnet_transform(pil_image).to(device)
                i = i + 1
                if label != 'unknown':
                    objects_in_image.append(img_tensor)
                    object_labels.append(label)
            else:
                img_tensor = segmentor.resnet_transform(pil_image).to(device)
                objects_in_image.append(img_tensor)

    with torch.no_grad():
        tensor_all_objs = torch.stack(objects_in_image).to(device)
        emb = segmentor.pass_through_resnet(tensor_all_objs)

    return emb, object_labels


# Action names from MiniGrid
ACTION_TYPES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
OBJECT_TYPES = ["empty", "wall", "door", "key", "ball", "box", "goal"]
INVENTORY_TYPES = ["none", "key", "ball"]

def one_hot(value, categories):
    vec = np.zeros(len(categories))
    if value in categories:
        vec[categories.index(value)] = 1
    return vec

def build_action_repr(action_str, object_ahead, inventory_item, success_flag=False):
    """
    Returns a vector encoding for the current action situation.
    """

    action_vec = one_hot(action_str, ACTION_TYPES)
    object_vec = one_hot(object_ahead, OBJECT_TYPES)
    inventory_vec = one_hot(inventory_item, INVENTORY_TYPES)
    relation_vec = np.array([1])  # always facing in MiniGrid
    success_vec = np.array([1 if success_flag else 0])

    return np.concatenate([action_vec, object_vec, inventory_vec, relation_vec, success_vec])

def compute_effect(st, st_plus_1, reward):
    difference = st_plus_1 - st
    combined_effect = torch.cat([difference.squeeze(0), torch.tensor(reward).unsqueeze(0).to(device)])
    return combined_effect

def build_action_repr_dict(action_str, object_ahead, inventory_item, success_flag=False):
    return {
        "action": action_str,
        "object": object_ahead,
        "inventory": inventory_item,
        "relation": "facing",
        "success": success_flag,
        "template": build_action_repr(action_str, object_ahead, inventory_item, success_flag).tolist()
    }



# Task runner
import random
def collect_minigrid_episodes():

    # model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    # "MiniGrid-Empty-5x5-v0",
    # "MiniGrid-DoorKey-5x5-v0",
    tasks = [

        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-SimpleCrossingS9N1-v0"
    ]
    gs = GraphStore("bolt://localhost:7687", "neo4j", "minigrid123")

    for task in tasks:
        env = gym.make(task, render_mode="rgb_array")

        env = FullyObsWrapper(env)
        seed = random.randint(0, 10000)
        done_true = 0
        done_false = 0
        total = 20

        # env = gym.make(task, render_mode='rgb_array')
        while done_true < 10 or done_false < 10:  # Collect 20 episodes
            mem_oracle = SmarterOracle()
            obs, _ = env.reset(seed=seed)

            env.unwrapped.agent_pos = (random.randint(1, 3), random.randint(1, 3))
            env.unwrapped.agent_dir = random.randint(0, 3)
            # ✅ Sync visible agent position + direction
            env.agent_pos = env.unwrapped.agent_pos
            env.agent_dir = env.unwrapped.agent_dir
            obs = env.gen_obs()

            episode_id = gs.add_episode(success=False, task=task)
            done = False
            t = 0
            total_reward = 0
            crt_frame = env.render()

            tensor_img_reduced = segmentor.resnet_transform_frame(crt_frame).float().to(device)
            encoded_state_t = segmentor.pass_through_resnet(tensor_img_reduced.unsqueeze(0))
            all_obj = []
            while not done and t <= 9:
                obs_visual = crt_frame
                # plt.imshow(crt_frame)
                # plt.show()

                action = mem_oracle.get_action(obs)
                # action = env.action_space.sample()



                direction = obs['direction']
                obs_mission_textual = obs['mission']
                # if t == 0:
                #     embedding = model.encode(obs_mission_textual)

                state_id = gs.add_state(episode_id, timestep=t, embedding=encoded_state_t.squeeze(0).tolist())
                if t > 0:
                    gs.add_edge(
                        source=aff_node_id,  # The affordance node
                        target=state_id,  # The resulting state
                        label="outcome",  # The causal effect
                        state_id=state_id,
                        source_tag="minigrid_data"
                    )

                emb_list, label_list = extract_and_embed_objects(obs_visual)
                emb_list = emb_list.tolist()
                i = 0
                for emb in emb_list:
                    if len(label_list) > 0:
                        label = label_list[i]
                    else:
                        label=None
                    similar_obj_ids, all_obj = cs.find_similar_object_concepts(emb,use_label=False, check_label=label, fetch_obj=all_obj)
                    if similar_obj_ids:
                        for sid in similar_obj_ids:
                            gs.add_edge(
                                source=state_id,
                                target=sid,
                                label="has_object",
                                weight=1.0,
                                state_id=state_id,
                            )
                    else:
                        id_obj = gs.add_object_concept(state_id, emb, label)
                        all_obj.append((id_obj, emb))
                    i=i+1
                    action_name = MINIGRID_ACTIONS[action]  # 'toggle'

                obs, reward, done, truncated, info = env.step(action)

                frame_t_1 = env.render()# ← RGB image from this step
                crt_frame =frame_t_1
                total_reward += reward
                tensor_img_reduced_t_1 = segmentor.resnet_transform_frame(frame_t_1).float().to(device)
                encoded_state_t_1 = segmentor.pass_through_resnet(tensor_img_reduced_t_1.unsqueeze(0))

                effect = compute_effect(encoded_state_t, encoded_state_t_1, reward)

                aff_label = f"{action_name}"
                aff_node_id = gs.create_node("Affordance", {
                        "label": aff_label,
                        "action": action_name,
                        "reward": reward,
                        'outcome': effect.tolist()
                    })
                encoded_state_t = encoded_state_t_1

                gs.add_edge(state_id, aff_node_id, label="influences", source_tag="minigrid")

                t =t+ 1

            # Mark episode success
            if done and total_reward >0:
                print('succ')
                done_true += 1
                with gs.driver.session() as session:
                    session.run(
                        "USE ltm3 MATCH (e:Episode) WHERE elementId(e) = $eid "
                        f"SET e.succesfull_outcome = true",
                        eid=episode_id)
            else:
                print('fail')
                done_false+= 1
                with gs.driver.session() as session:
                        session.run(
                            "USE ltm3 MATCH (e:Episode) WHERE elementId(e) = $eid "
                            f"SET e.succesfull_outcome = false",
                            eid=episode_id
                    )
                    # session.run(
                    #     "USE ltm MATCH (e:Episode) WHERE elementId(e) = $eid "
                    #     f"SET e.succesfull_outcome = true SET e.mission_embedding=$mission_embedding",
                    #     eid=episode_id,
                    #     mission_embedding=embedding
                    # )

    gs.close()

collect_minigrid_episodes()
