import torch
import os
import io
import gym
import random
import wandb
import PIL
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_image
from itertools import count

from create.create_game.settings import CreateGameSettings


from hetgraph_gt_encoder.data_helpers.data_preparation import StateLoader
from hetgraph_gt_encoder.models.HeCo import HeCo
from hetgraph_gt_encoder.heco_params import heco_params
from baseline_models.logger import Logger

from co_segment_anything.sam_utils import SegmentAnythingObjectExtractor
from create.create_game.settings import CreateGameSettings
from memory_graph.gds_concept_space import ConceptSpaceGDS
from memory_graph.memory_utils import WorkingMemory
from affordance_learning.action_observation.utils import ActionObservation


st_loader = StateLoader(nr_mps=2, mps=None, use_memory='afftest')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GTEncoder:

    def __init__(self, checkpoint_path, use_logger=True):
        self.object_extractor = SegmentAnythingObjectExtractor()
        self.action_embedder = ActionObservation()
        self.concept_space = ConceptSpaceGDS(memory_type="afftest")
        self.wm = WorkingMemory(which_db='afftest')
        self.env = self.load_env()
        self.model = self.load_model(checkpoint_path)
        if use_logger:
            # self.table = wandb.Table(columns=["intra_att", "type_lvl_att"])
            wandb_logger = Logger(f"eval_attention", project='graph_encoder')
            self.logger = wandb_logger.get_logger()

    def load_env(self):
        env = gym.make(f'CreateLevelPush-v0')
        settings = CreateGameSettings(
            evaluation_mode=True,
            max_num_steps=30,
            action_set_size=7,
            render_mega_res=False,
            render_ball_traces=False)
        env.set_settings(settings)
        env.reset()
        return env

    def load_model(self, checkpoint_path):

        (batch_pos1, batch_pos2, batch_neg1), key, all_aff_keys, all_obj_keys, action_keys, (
            fstate_p1, fstate_p2, fstate_n1), t = st_loader.get_subgraph_state_data(batch_size=1)
        feats = batch_pos1[0][0]
        nei_index = batch_pos1[0][1]
        mps = st_loader.generate_mps_st(nei_index, all_aff_keys, all_obj_keys, action_keys, fstate_p1)
        mps_dims = [mp.shape[1] for mp in mps]
        feats_dim_list = [i.shape[1] for i in batch_pos1[0][0]]
        count_mps = 2
        args = heco_params()
        model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                     count_mps, args.sample_rate, args.nei_num, args.tau, args.lam, mps_dims).to(device)
        model.load_state_dict(torch.load(checkpoint_path))

        model.eval()
        return model

    def compute_mask(self):
        sorter = np.argsort(self.env.allowed_actions)
        b = sorter[np.searchsorted(self.env.allowed_actions, self.env.inventory, sorter=sorter)]
        mask = np.zeros(self.env.allowed_actions.shape, dtype=bool)  # np.ones_like(a,dtype=bool)
        mask[b] = True
        return torch.tensor(mask), self.env.inventory

    def select_action(self, steps_done):
        random_action_from_inventory = np.where(self.env.inventory == random.choice(self.env.inventory))[0]
        mask, inventory = self.compute_mask()
        return [[random_action_from_inventory.item(), random.uniform(-1, 1),
                 random.uniform(-1, 1)]], steps_done, mask, torch.tensor(inventory).unsqueeze(0)

    def get_current_state_graph(self, observation, objects_interacting_frames,  episode_id, timestep, state_image=None):
        current_screen_objects, encoded_state, masks = self.object_extractor.extract_objects(observation)
        state_id, added_objs = self.wm.add_to_memory(encoded_state, current_screen_objects, episode_id, timestep, masks)
        action_tool_ids, added_objs_act = self.wm.add_object_action_repr(objects_interacting_frames, state_id)
        return current_screen_objects, encoded_state, state_id, action_tool_ids, added_objs, added_objs_act

    def compute_effect(self, st, st_plus_1, reward):
        difference = st_plus_1 - st
        combined_effect = torch.cat([difference.squeeze(0), torch.tensor(reward).unsqueeze(0).to(device)])
        return combined_effect

    def log_object_attention(self, intra_att, type_lvl_att, added_objs, added_objs_actions):
        fig = plot_inter_lvl_att(intra_att[0].tolist()[0])
        image_bytes = to_image(fig, format="png")

        heatmap_data_type = [go.Heatmap(z=[type_lvl_att], colorscale='Viridis')]
        heatmap_data_inter = [go.Heatmap(z=[intra_att[0].squeeze(0).squeeze(1).tolist()], colorscale='Viridis')]

        # Log the 1D heatmap to WandB
        wandb.log({"type lvl att": go.Figure(heatmap_data_type)})
        wandb.log({"inter obj att": go.Figure(heatmap_data_inter)})

        obj_images = []
        for obj in added_objs:
            obj_img = wandb.Image(self.wm.aws_utils.get_data(obj))
            obj_images.append(obj_img)

        wandb.log({"Objects": obj_images})

        obj_images_actions = []
        for obj in added_objs_actions:
            obj_img = wandb.Image(self.wm.aws_utils.get_data(obj))
            obj_images_actions.append(obj_img)

        wandb.log({"Objects actions": obj_images_actions})
        return

    def log_image(self, image_data, step):
        # Log the image to wandb
        wandb.log({"Image": [wandb.Image(image_data, caption=f'Time Step: {step}')]})

    def run(self, num_episodes=3):
        steps_done = 0
        alpha = 0.5
        loss_type = None
        for i_episode in range(num_episodes):
            print(f"Episode:{i_episode}")
            self.env.reset()
            concept_space_episode_id = self.concept_space.add_data('Episode')
            episode_id = concept_space_episode_id['elementId(n)'][0]
            self.concept_space.close()
            episode_memory = []
            obs = self.env.reset()
            rew_ep = 0
            loss_ep = 0
            timestep = 0
            aff_id = None
            state_id = None
            encoded_inventory, objects_interacting_frames = self.action_embedder.get_inventory_embeddings(self.env.inventory, self.object_extractor)
            current_screen, encoded_state_t, state_id, action_tool_ids, added_objs, added_objs_actions = self.get_current_state_graph(
                obs, objects_interacting_frames,
                episode_id, timestep
            )
            self.log_image(obs, timestep)
            feats, nei_index, mps = st_loader.get_state_data_by_id(state_id)
            z_sc, z_mp, intra_att, type_lvl_att = self.model(feats, nei_index, mps, alpha, loss_type, testing=True)
            self.log_object_attention(intra_att, type_lvl_att, added_objs, added_objs_actions)
            self.wm.concept_space.match_state_add_encs(state_id, z_sc.tolist(), z_mp.tolist())
            for t in count():
                # Select and perform an action
                action, steps_done, mask, inventory = self.select_action(steps_done)

                returned_state, reward, done, _ = self.env.step(action[0])
                inventory_item_applied = self.env.inventory.item(action[0][0])
                position_applied = [action[0][1], action[0][2]]
                try:
                    r = reward.item()
                except:
                    r = reward
                    print(reward)

                aff_id = self.wm.match_action_affordance(action_tool_ids, inventory_item_applied, position_applied, r, timestep)
                self.wm.concept_space.match_state_add_aff(state_id, aff_id)

                rew_ep += r

                current_screen, encoded_state_t_plus_1, state_id, action_tool_ids,  added_objs, added_objs_actions = self.get_current_state_graph(
                    returned_state, objects_interacting_frames,
                    episode_id, timestep+1
                )
                affordance_effect = self.compute_effect(encoded_state_t, encoded_state_t_plus_1, r)
                episode_memory.append(current_screen)
                self.log_image(returned_state, timestep+1)
                feats, nei_index, mps = st_loader.get_state_data_by_id(state_id)
                z_sc, z_mp, intra_att, type_lvl_att = self.model(feats, nei_index, mps, alpha, loss_type, testing=True)
                self.log_object_attention(intra_att, type_lvl_att, added_objs, added_objs_actions)
                self.wm.concept_space.match_state_add_encs(state_id, z_sc.tolist(), z_mp.tolist())

                if not done:
                    next_state = current_screen
                else:
                    next_state = None

                # Move to the next state
                state = next_state

                timestep += 1
                if aff_id is not None and state_id is not None:
                    self.wm.concept_space.match_state_add_aff_outcome(state_id, aff_id)
                    self.wm.concept_space.set_property(aff_id, 'Affordance', 'outcome', affordance_effect.tolist())

        return


    def view_objects_state(self, state_id):
        import matplotlib.pyplot as plt
        states = self.wm.concept_space.get_objects_for_state(state_id)
        for idx, state in states.iterrows():
            img = self.wm.aws_utils.get_data(state["elementId(o)"])
            plt.imshow(img)
            plt.show()

        return


def get_data_for_state(st_loader, batch, all_aff_keys, all_obj_keys, action_keys, full_state):
    feats = batch[0][0]
    nei_index = batch[0][1]
    mps = st_loader.generate_mps_st(nei_index, all_aff_keys, all_obj_keys, action_keys, full_state)
    return feats, nei_index, mps

def plot_inter_lvl_att(data):
    fig = px.imshow(data, color_continuous_scale="Viridis")
    # Customize the layout
    fig.update_layout(
        title="1D Heatmap",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
    )
    return fig


def load_attention_wandb():

    table = wandb.Table(columns=["intra_att", "type_lvl_att"])
    wandb_logger = Logger(f"eval_attention", project='graph_encoder')
    logger = wandb_logger.get_logger()

    PATH_TO_CHECKPOINTS = os.path.join(str(os.getcwd()), "checkpoints")
    checkpoint_files = ["triplet_0.5_8_3.197206497192383.pkl", "triplet_loss_0.2_8_0.22625350952148438.pkl",
                        "contrastive_0.5_9_1.4001202583312988.pkl"]
    checkpoint_path = os.path.join(PATH_TO_CHECKPOINTS, checkpoint_files[0])
    gt_encoder = GTEncoder(checkpoint_path)
    model = gt_encoder.model

    alpha = 0.5
    loss_type = None
    for test in range(10):
        (batch_pos1, batch_pos2, batch_neg1), key, all_aff_keys, all_obj_keys, action_keys, (
            fstate_p1, fstate_p2, fstate_n1), (tp1, tp2, tn) = st_loader.get_subgraph_state_data(batch_size=1)
        feats_p1, nei_index_p1, mps_p1 = get_data_for_state(st_loader, batch_pos1, all_aff_keys, all_obj_keys,
                                                            action_keys, fstate_p1)

        z_sc, z_mp, intra_att, type_lvl_att = model(feats_p1, nei_index_p1, mps_p1, alpha, loss_type, testing=True)

        fig = plot_inter_lvl_att(intra_att[0].tolist()[0])
        image_bytes = to_image(fig, format="png")

        heatmap_data_type = [go.Heatmap(z=[type_lvl_att], colorscale='Viridis')]
        heatmap_data_inter = [go.Heatmap(z=[intra_att[0].squeeze(0).squeeze(1).tolist()], colorscale='Viridis')]

        # Log the 1D heatmap to WandB
        wandb.log({"type lvl att": go.Figure(heatmap_data_type)})
        wandb.log({"inter obj att": go.Figure(heatmap_data_inter)})


def main():
    #load_attention_wandb()
    PATH_TO_CHECKPOINTS = os.path.join(str(os.getcwd()), "checkpoints")
    checkpoint_files = ["triplet_0.5_8_3.197206497192383.pkl", "triplet_loss_0.2_8_0.22625350952148438.pkl",
                        "contrastive_0.5_9_1.4001202583312988.pkl"]
    checkpoint_path = os.path.join(PATH_TO_CHECKPOINTS, checkpoint_files[0])
    gt_encoder = GTEncoder(checkpoint_path)
    gt_encoder.view_objects_state("4:c3295efd-cd8c-4d8a-8839-0d538258dc83:12")
    # gt_encoder.run()




if __name__ == '__main__':
    main()




