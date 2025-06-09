import gym
from PIL import Image
import matplotlib.pyplot as plt
import json
import time
from create.create_game.settings import CreateGameSettings
from affordance_learning.action_observation.utils import ActionObservation
from co_segment_anything.sam_utils import SegmentAnythingObjectExtractor
from run_rl_with_obj_action_repr import TrainModel
from baseline_models.logger import Logger
from co_segment_anything.dqn_sam import DQN
from datetime import datetime as dt


import os
import pickle

def save_episodes_states_visually(obs, ep_id, st):
    image_pil = Image.fromarray(obs)
    dir = os.getcwd()
    ep_dir = os.path.join(dir, 'episodes')
    ep_dir = os.path.join(ep_dir, f"{ep_id.split(':')[2]}")
    if not os.path.exists(ep_dir):
        os.makedirs(ep_dir)
    file_name = f'st_{st}.png'
    file_path = os.path.join(ep_dir, file_name)

    image_pil.save(file_path)
    return
def main():

    tasks = [ "CreateLevelPush","CreateLevelBuckets","CreateLevelBasket", "CreateLevelBelt","CreateLevelObstacle"]
    # tasks = [  "CreateLevelPush","CreateLevelBuckets", "CreateLevelPush","CreateLevelBuckets", "CreateLevelBasket","CreateLevelCollide", "CreateLevelMoving", "CreateLevelBelt", "CreateLevelNavigate", "CreateLevelObstacle"]
    succesfull_episodes = {
        "CreateLevelBasket-v0":[],
        "CreateLevelBuckets-v0":[],
        "CreateLevelCollide-v0":[],
        "CreateLevelMoving-v0":[],
        "CreateLevelBelt-v0":[],
        "CreateLevelPush-v0":[],
        "CreateLevelNavigate-v0":[],
        "CreateLevelObstacle-v0":[]
    }
    action_embedder = ActionObservation()
    object_extractor = SegmentAnythingObjectExtractor()
    max_steps = 9
    for task in tasks:
        done = False
        task_i = f'{task}-v0'
        env = gym.make(f'{task}-v0')
        settings = CreateGameSettings(
            evaluation_mode=True,
            max_num_steps=max_steps,
            action_set_size=5,
            render_mega_res=False,
            render_ball_traces=False)
        trainer = TrainModel(DQN,
                             env)
        env.set_settings(settings)

        env.reset()

        env.set_task_id(task_i)
        count_outcome = 0
        nr_outcome = 50
        if task == "CreateLevelBasket":
            nr_outcome = 11
        max_iter = 0
        while count_outcome < nr_outcome or max_iter == 5000:
            obs = env.reset()

            done = False
            actions_taken_in_episode = []

            while not done and len(actions_taken_in_episode) < max_steps:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                actions_taken_in_episode.append(action)
            print(len(actions_taken_in_episode))


            print(f"{task_i} {info['cur_goal_hit']}")
            max_iter += 1

            if info['cur_goal_hit'] == 1:
                dict_env = env.__dict__
                count_outcome = count_outcome + 1
                concept_space_episode_id = trainer.concept_space.add_data('Episode')
                episode_id = concept_space_episode_id['elementId(n)'][0]
                success_inventory = env.inventory
                trainer.concept_space.set_property(episode_id, 'Episode', 'succesfull_outcome', True)
                trainer.concept_space.set_property(episode_id, 'Episode', 'task', task_i, is_string=True)
                trainer.concept_space.set_property(episode_id, 'Episode', 'inventory',
                                                   ",".join(map(str, list(success_inventory))), is_string=True)
                trainer.concept_space.set_property(episode_id, 'Episode', 'env_pos',
                                                   json.dumps([list(s) for s in list(env.env_pos)]),
                                                   is_string=True)
                successful_steps_compatible = [[int(step[0]), float(step[1]), float(step[2])] for step in
                                               actions_taken_in_episode]

                trainer.concept_space.set_property(episode_id, 'Episode', 'actions_taken',
                                                   json.dumps(successful_steps_compatible), is_string=True)
                for key in dict_env.keys():
                    try:
                        if key != 'actions_taken' and key != 'env_pos':
                            trainer.concept_space.set_property(episode_id, 'Episode', str(key), json.dumps(dict_env[key]))
                    except:
                        print(key)




                trainer.concept_space.close()
                timestep = 0

                obs=env.reset()
                env.inventory = success_inventory

                encoded_inventory, objects_interacting_frames = trainer.action_embedder.get_inventory_embeddings(
                env.inventory, object_extractor, wm=trainer.wm)

                current_screen, encoded_state_t, state_id, action_tool_ids = trainer.get_current_state_graph(
                obs, objects_interacting_frames,
                    episode_id, timestep
                    )

                new_done = False
                st = 0
                for action in actions_taken_in_episode:
                    action_t = action[0]
                    pos1 = action[1]
                    pos2 = action[2]
                    if not new_done:
                        obs, reward, new_done, info = env.step(action)
                        save_episodes_states_visually(obs, episode_id, st)


                        inventory_item_applied = env.inventory.item(action_t)
                        position_applied = [pos1, pos2]

                        # encoded_inventory, objects_interacting_frames = action_embedder.get_inventory_embeddings(
                        # env.inventory, object_extractor)

                        aff_id = trainer.wm.match_action_affordance(action_tool_ids, inventory_item_applied,
                                                                position_applied, reward,
                                                                timestep)
                        trainer.wm.concept_space.match_state_add_aff(state_id, aff_id)


                        current_screen, encoded_state_t_plus_1, state_id, action_tool_ids = trainer.get_current_state_graph(
                            obs, objects_interacting_frames,
                            episode_id, timestep + 1
                        )


                        affordance_effect = trainer.compute_effect(encoded_state_t, encoded_state_t_plus_1, reward)
                        encoded_state_t = encoded_state_t_plus_1

                        timestep += 1
                        if aff_id is not None and state_id is not None:
                            trainer.wm.concept_space.match_state_add_aff_outcome(state_id, aff_id)
                            trainer.wm.concept_space.set_property(aff_id, 'Affordance', 'outcome', affordance_effect.tolist())
                        st += 1
            else:
                dict_env = env.__dict__
                count_outcome = count_outcome + 1
                concept_space_episode_id = trainer.concept_space.add_data('Episode')
                episode_id = concept_space_episode_id['elementId(n)'][0]
                success_inventory = env.inventory
                trainer.concept_space.set_property(episode_id, 'Episode', 'succesfull_outcome', False)
                trainer.concept_space.set_property(episode_id, 'Episode', 'task', task_i, is_string=True)
                trainer.concept_space.set_property(episode_id, 'Episode', 'inventory',
                                                   ",".join(map(str, list(success_inventory))), is_string=True)
                trainer.concept_space.set_property(episode_id, 'Episode', 'env_pos',
                                                   json.dumps([list(s) for s in list(env.env_pos)]),
                                                   is_string=True)
                successful_steps_compatible = [[int(step[0]), float(step[1]), float(step[2])] for step in
                                               actions_taken_in_episode]

                trainer.concept_space.set_property(episode_id, 'Episode', 'actions_taken',
                                                   json.dumps(successful_steps_compatible), is_string=True)
                for key in dict_env.keys():
                    try:
                        if key != 'actions_taken' and key != 'env_pos':
                            trainer.concept_space.set_property(episode_id, 'Episode', str(key),
                                                               json.dumps(dict_env[key]))
                    except:
                        print(key)

                trainer.concept_space.close()
                timestep = 0

                obs = env.reset()
                env.inventory = success_inventory
                invent_time = dt.now()
                encoded_inventory, objects_interacting_frames = trainer.action_embedder.get_inventory_embeddings(
                    env.inventory, object_extractor, wm=trainer.wm)
                inv_time_end = dt.now()
                duration = invent_time - inv_time_end
                duration_in_seconds = duration.total_seconds()
                print(f"Duration in seconds: {duration_in_seconds} seconds")

                current_screen, encoded_state_t, state_id, action_tool_ids = trainer.get_current_state_graph(
                    obs, objects_interacting_frames,
                    episode_id, timestep
                )

                new_done = False
                st = 0
                for action in actions_taken_in_episode:
                    action_t = action[0]
                    pos1 = action[1]
                    pos2 = action[2]
                    if not new_done:
                        obs, reward, new_done, info = env.step(action)
                        save_episodes_states_visually(obs, episode_id, st)

                        inventory_item_applied = env.inventory.item(action_t)
                        position_applied = [pos1, pos2]

                        # encoded_inventory, objects_interacting_frames = action_embedder.get_inventory_embeddings(
                        #     env.inventory, object_extractor)

                        aff_id = trainer.wm.match_action_affordance(action_tool_ids, inventory_item_applied,
                                                                    position_applied, reward,
                                                                    timestep)
                        trainer.wm.concept_space.match_state_add_aff(state_id, aff_id)

                        current_screen, encoded_state_t_plus_1, state_id, action_tool_ids = trainer.get_current_state_graph(
                            obs, objects_interacting_frames,
                            episode_id, timestep + 1
                        )

                        affordance_effect = trainer.compute_effect(encoded_state_t, encoded_state_t_plus_1, reward)
                        encoded_state_t = encoded_state_t_plus_1

                        timestep += 1
                        if aff_id is not None and state_id is not None:
                            trainer.wm.concept_space.match_state_add_aff_outcome(state_id, aff_id)
                            trainer.wm.concept_space.set_property(aff_id, 'Affordance', 'outcome',
                                                                  affordance_effect.tolist())
                        st += 1
            print(info['cur_goal_hit'])






                # try:
                #     env.render('human')
                # except:
                #     pass

    with open('success_ep.pickle', 'wb') as handle:
        pickle.dump(succesfull_episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)


        # while not done:
        #     obs, reward, done, info = env.step(env.action_space.sample())
        #     env.render('human')

        # frames, reward, hit_goal, dict_r = env.step([random.randint(0, 6)])
        # plt.imshow(frames)
        # plt.show()


if __name__ == '__main__':
    main()