import random
import wandb
import gym
import sys
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable

IS_SERVER = True
from baseline_models.logger import Logger
from co_segment_anything.dqn_sam import DQN
from co_segment_anything.sam_utils import SegmentAnythingObjectExtractor
from create.create_game.settings import CreateGameSettings
from memory_graph.gds_concept_space import ConceptSpaceGDS
from memory_graph.memory_utils import WorkingMemory

envs_to_run = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','mask', 'inventory'))


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
        self.concept_space = ConceptSpaceGDS(memory_type="wmtest")
        self.wm = WorkingMemory()

    def train(self, target_net, policy_net, memory, params, optimizer, writer, max_timesteps=30):
        episode_durations = []
        num_episodes = 3000
        steps_done = 0
        counter = 0
        smallest_loss = 99999
        loss = 0
        for i_episode in range(num_episodes):
            print(f"Episode:{i_episode}")
            concept_space_episode_id = self.concept_space.add_data('Episode')
            episode_id = concept_space_episode_id['elementId(n)'][0]
            self.concept_space.close()
            episode_memory = []
            # Initialize the environment and state
            obs = self.env.reset()
            #state = self.process_frames(obs)
            state, encoded_state = self.object_extractor.extract_objects(obs)
            rew_ep = 0
            loss_ep = 0
            losses = []
            timestep = 0
            for t in count():
                # Select and perform an action
                timestep += 1
                action, steps_done, mask, inventory = self.select_action(state, params, policy_net, len(self.env.allowed_actions), steps_done)

                returned_state, reward, done, _ = self.env.step(action[0])
                self.writer.log({"Action taken": action[0][0]})
                reward = torch.tensor([reward], device=device)

                rew_ep += reward.item()
                #current_screen = self.process_frames(screen)
                current_screen, encoded_state = self.object_extractor.extract_objects(returned_state)
                # episode_memory.append(current_screen)
                # ids = self.concept_space.add_data('ObjectConcept')
                # self.concept_space.update_node_by_id(ids['elementId(n)'][0],current_screen.squeeze(0).squeeze(0)[0].tolist())
                #self.concept_space.add_state_with_objects(encoded_state, current_screen, episode_id, timestep)
                self.wm.add_to_memory(encoded_state, current_screen, episode_id, timestep, reward)
                if not done:
                    next_state = current_screen
                else:
                    next_state = None

                # Store the transition in memory
                self.wm.compute_attention(timestep, episode_id)
                if next_state is not None:
                    memory.push(state, action[0], next_state, reward, mask, inventory)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss_ep = self.optimize_model(policy_net, target_net, params, memory, optimizer, loss_ep, writer)

                if done or t == max_timesteps:
                    episode_durations.append(t + 1)
                    self.writer.log(
                        {"Reward episode": rew_ep, "Episode duration": t + 1, "Train loss": loss_ep / (t + 1)})
                    # print(loss_ep / (t + 1))
                    # episode_frames_wandb = make_grid(episode_frames)
                    # images = wandb.Image(episode_frames_wandb, caption=f'Episode {i_episode} states')
                    # self.writer.log({'states': episode_frames})

                    break
                # Update the target network, copying all weights and biases in DQN
            if i_episode % params['target_update'] == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if i_episode % 150 == 0 and i_episode != 0:
                self.evaluate(target_net, writer, i_episode)
            if i_episode % 100 == 0 and i_episode != 0:
                PATH = f"model_{i_episode}_{loss_ep}.pt"
                torch.save({
                    'epoch': i_episode,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_ep,
                }, PATH)
        return

    def compute_td_loss(self, model, replay_buffer, params, optimizer, batch_size=32):

        state, action, reward, next_state, done, non_final_mask = replay_buffer.sample_td(batch_size)
        # state = self.object_extractor.extract_objects(state)
        # next_state = self.object_extractor.extract_objects(next_state)

        state = state.to(device)
        next_state = next_state.to(device)
        action = action.to(device)

        target = action.squeeze(1)
        values = torch.tensor(self.ACTIONS_TO_USE).to(device)
        t_size = target.numel()
        v_size = values.numel()
        t_expand = target.unsqueeze(1).expand(t_size, v_size)
        v_expand = values.unsqueeze(0).expand(t_size, v_size)
        result_actions = (t_expand - v_expand == 0).nonzero()[:, 1].unsqueeze(1)

        reward = reward.to(device)
        done = done.to(device)

        q_values = model(state)
        #q_values = torch.tensor([[self.ACTIONS_TO_USE[model(state).max(1)[1].view(1, 1)]]])
        next_q_values = torch.zeros(batch_size, device=device)
        next_q_values[non_final_mask] = model(next_state).max(1)[0]

        q_value = q_values.gather(1, result_actions).squeeze(1)

        expected_q_value = reward + params['gamma'] * next_q_values * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def init_model(self, actions=0, checkpoint_file=""):
        obs = self.env.reset()
        init_screen, state_enc = self.object_extractor.extract_objects(obs)
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
            self.train(target_net, policy_net, self.memory, self.params, optimizer, self.writer)

        return

    def process_frames(self,obs):
        resize = T.Compose([T.ToPILImage(),
                            T.Resize(128, interpolation=Image.CUBIC),
                            T.ToTensor()])
        screen = obs.transpose((2, 0, 1))
        _, screen_height, screen_width = screen.shape
        screen = torch.tensor(screen)
        return resize(screen).unsqueeze(0).to(device)

    def compute_mask(self):
        sorter = np.argsort(self.env.allowed_actions)
        b = sorter[np.searchsorted(self.env.allowed_actions, self.env.inventory, sorter=sorter)]
        mask = np.zeros(self.env.allowed_actions.shape, dtype=bool)  # np.ones_like(a,dtype=bool)
        mask[b] = True

        return torch.tensor(mask, device=device), self.env.inventory

    def select_action(self, state, params, policy_net, n_actions, steps_done):
        sample = random.random()
        eps_threshold = 0.8
        # eps_threshold = params['eps_end'] + (params['eps_start'] - params['eps_end']) * \
        #     math.exp(-1. * steps_done / params['eps_decay'])
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #Exploiting
                mask, inventory = self.compute_mask()
                q_vals_discrete, q_val_cont, action_sel = policy_net(state, mask, torch.tensor(inventory).unsqueeze(0))
                return action_sel, steps_done, mask, torch.tensor(inventory).unsqueeze(0)
        else:
            random_action_from_inventory = np.where(self.env.inventory == random.choice(self.env.inventory))[0]
            mask, inventory = self.compute_mask()
            return [[random_action_from_inventory.item(), random.uniform(-1,1), random.uniform(-1, 1)]], steps_done, mask, torch.tensor(inventory).unsqueeze(0)

    def optimize_model(self, policy_net, target_net, params, memory, optimizer, loss_ep, writer):
        if len(memory) < params['batch_size']:
            return loss_ep

        transitions = memory.sample(params['batch_size'])
        batch = Transition(*zip(*transitions))

        # Convert the batch of transitions to tensors
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.tensor(batch.action).float().to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        reward_batch = torch.cat(batch.reward).float().to(device)
        mask_batch = torch.stack(batch.mask)
        inventory_batch = torch.stack(batch.inventory)

        # Separate the categorical and continuous actions in the action batch
        num_categorical_actions = policy_net.num_discrete_actions
        categorical_actions = action_batch[:, :num_categorical_actions]
        continuous_actions = action_batch[:, num_categorical_actions:]

        # Compute the Q values for the current state-action pairs using the policy network
        q_values_categorical, q_values_continuous, actions = policy_net(state_batch, mask_batch, inventory_batch)
        actions = torch.tensor(actions, device=device, requires_grad=True, dtype=torch.float32)
        # state_action_values = q_values_categorical.gather(1, categorical_actions.long()).squeeze()
        # state_action_values_continuous = torch.sum(q_values_continuous * continuous_actions, dim=1)

        # Compute the Q values for the next states using the target network

        next_q_values_categorical, next_q_values_continuous, actions_next_state = target_net(next_state_batch,
                                                                                             mask_batch,
                                                                                             inventory_batch)
        actions_next_state = torch.tensor(actions_next_state, device=device, requires_grad=True, dtype=torch.float32)

        # # Compute the maximum Q values for the next states (used for the target values)
        # next_state_values, _ = torch.max(next_q_values_categorical, dim=1)
        # next_state_values_continuous = torch.sum(next_q_values_continuous * continuous_actions, dim=1)
        #
        # # Compute the target Q values using the Bellman equation
        expected_state_action_values = reward_batch + params["gamma"] * actions_next_state[:, 0]
        expected_state_action_values_continuous = reward_batch + params["gamma"] * actions_next_state[:, 1:].sum(dim=1)

        # Calculate the categorical loss (Cross-Entropy Loss)
        categorical_loss = F.smooth_l1_loss(actions[:, 0], expected_state_action_values)

        # Calculate the continuous loss (Mean Squared Error)
        continuous_loss = F.mse_loss(actions[:, 1:].sum(dim=1), expected_state_action_values_continuous)

        # Calculate the total loss (sum of categorical and continuous losses)
        total_loss = categorical_loss + continuous_loss
        writer.log({'Batch reward': reward_batch.sum().output_nr})
        # Optimize the policy network
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss.item()

    def evaluate(self, target_net, writer, i_episode, max_timesteps=1000):
        with torch.no_grad():
            # Initialize the environment and state
            obs = self.env.reset()
            # last_screen = self.process_frames()
            current_screen = self.process_frames(obs.copy())
            state = current_screen
            rew_ep = 0
            for t in count():
                mask, inventory = self.compute_mask()
                q_vals_discrete, q_val_cont, action_sel = target_net(state, mask, torch.tensor(inventory).unsqueeze(0))
                screen, reward, done, _ = self.env.step(action_sel[0])
                reward = torch.tensor([reward], device=device, dtype=torch.float32)
                rew_ep += reward.item()
                state = self.process_frames(screen)
                if done or t == max_timesteps:
                    writer.log({"Reward episode test": rew_ep})
                    break
        return


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


def process_frames_a(state):
    resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(),
                        T.ToTensor()])
    screen = torch.tensor(state).T
    return resize(screen).to(device)


def main():
    args = sys.argv[1:]
    checkpoint_file = ""
    if len(args) >0 and args[0] == '-checkpoint':
        checkpoint_file = args[1]
    params = {
        'batch_size': 10,
        'gamma': 0.99,
        'eps_start': 0.9,
        'eps_end': 0.02,
        'eps_decay': .999985,
        'target_update': 1000
    }

    env = gym.make(f'CreateLevelPush-v0')
    settings = CreateGameSettings(
        evaluation_mode=True,
        max_num_steps=30,
        render_mega_res=False,
        render_ball_traces=False)
    env.set_settings(settings)
    env.reset()
    done = False
    frames = []

    wandb_logger = Logger(f"6_obj{checkpoint_file}samobjects_dqn_create", project='memory_testing')
    logger = wandb_logger.get_logger()
    trainer = TrainModel(DQN,
                         env, (True, 1000),
                         logger,True, params)
    trainer.init_model(checkpoint_file=checkpoint_file)


    return

if __name__ == '__main__':
    main()