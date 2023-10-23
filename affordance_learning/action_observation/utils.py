import random
import os
import gym
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from gym.envs.registration import register
from affordance_learning.action_observation.args import get_args
from affordance_learning.action_observation.env_interface import register_env_interface
from affordance_learning.action_observation.create_env_interface import CreateGameInterface, CreatePlayInterface

from affordance_learning.affordance_trained import ActionEmbedder

path_ds_aff = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "create_aff_ds\\objects_obs")
register_env_interface('^Create((?!Play).)*$', CreateGameInterface)
register_env_interface('^Create(.*?)Play(.*)?$', CreatePlayInterface)
register_env_interface('^StateCreate(.*?)Play(.*)?$', CreatePlayInterface)

register(
    id='CreateGamePlay-v0',
    entry_point='affordance_learning.action_observation.create_play:CreatePlay',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActionObservation:

    def __init__(self, concept_space=None):
        env = gym.make(f'CreateGamePlay-v0')
        args = get_args()
        env.update_args(args)
        self.concept_space = concept_space
        self.action_embedder = ActionEmbedder(concept_space)
        self.env = env
        self.transforms_img = transforms.Compose(
            [
                transforms.Resize(64, interpolation=Image.Resampling.BICUBIC),
                transforms.PILToTensor(),
            ]
        )
        return

    def get_observation(self, tool_id):
        obs, reward, done = self.env.step(tool_id)
        self.env.reset()
        all_obs = []
        for img in list(obs):
            im = Image.fromarray((img * 255).astype('uint8'))
            img_tensor = self.transforms_img(im)
            all_obs.append(img_tensor)

        all_obs = torch.stack(all_obs).to(device)
        all_obs = all_obs.type(torch.float32)

        return obs, all_obs

    def get_action_embedding(self, tool_id):
        tool_type = self.env.ALL_TOOLS[tool_id].tool_type

        if tool_type != 'no_op':
            data_numpy, data_tensor = self.get_observation(tool_id)
            context_latent, obj_instance, recon_img = self.action_embedder.get_action_embedding(data_tensor)
            if self.concept_space is not None:
                self.action_embedder.add_action_to_concept_space(context_latent,tool_type)
            return obj_instance[3], data_numpy[0], tool_type
        return None, None, None

    def get_single_inventory_emb(self, tool_id, object_model):
        context_action, img_interaction, tool_type = self.get_action_embedding(tool_id)
        if context_action is not None:
            img_interaction_pil = (img_interaction * 255).astype('uint8')
            objects_interacting, full_encodings = object_model.extract_objects(img_interaction_pil)
            interaction = {
                'objects_in_interaction': objects_interacting,
                'interaction': context_action,
                'context': full_encodings,
                'tool_label': tool_type,
                'tool_id': tool_id
            }
            return context_action, interaction
        return None, None

    def get_inventory_embeddings(self, inventory, object_model):
        all_repr = []
        all_interaction_snapshots = []

        for tool_id in inventory:
            context_action, interaction = self.get_single_inventory_emb(tool_id, object_model)
            if context_action is not None:
                all_repr.append(context_action)
                all_interaction_snapshots.append(interaction)

        return torch.stack(all_repr).squeeze(1), all_interaction_snapshots



