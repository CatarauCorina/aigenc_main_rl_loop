import os

import gym
import sys
import numpy as np
import random
from collections import namedtuple
from itertools import count
from PIL import Image
import matplotlib.pyplot as plt
from create.create_game.settings import CreateGameSettings
from affordance_learning.action_observation.utils import ActionObservation
from co_segment_anything.sam_utils import SegmentAnythingObjectExtractor
from run_rl_with_obj_action_repr import TrainModel
from baseline_models.logger import Logger
from co_segment_anything.dqn_sam import DQN



import pickle

def set_attributes_from_dict(env, data):
    for key, value in data.items():
        if hasattr(env, key) and key != 'done':
            if isinstance(value, list):
                # Check if the list contains lists (i.e., is a nested list)
                if all(isinstance(i, list) for i in value):
                    # Convert nested lists to NumPy arrays
                    setattr(env, key, np.array(value))
                else:
                    setattr(env, key, value)
            else:
                setattr(env, key, value)
def main():

    tasks = ["CreateLevelPush"]
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

    succesfull_episodes = [[1, -0.11074898391962051, 0.4193184971809387], [2, 0.33829265832901, 0.4043590724468231], [5, 0.3739112615585327, -0.7538906335830688], [5, 0.6918389201164246, 0.13065263628959656], [0, -0.8956891894340515, 0.845088541507721], [5, -0.2510870397090912, 0.44813793897628784], [3, -0.1892688125371933, -0.44736313819885254], [0, -0.9697389602661133, 0.7000598907470703], [3, 0.8464118242263794, -0.5620909929275513], [5, 0.4316059648990631, -0.1675329953432083], [2, 0.5961813926696777, -0.2638380527496338], [2, -0.5156179666519165, 0.6259537935256958], [4, -0.19858503341674805, 0.2313361018896103], [2, 0.9166705012321472, -0.20035524666309357], [1, -0.694593608379364, 0.7903221845626831], [4, 0.6318910121917725, 0.3675384819507599], [1, -0.5717999339103699, 0.1635398119688034], [5, 0.4309033751487732, -0.6987014412879944], [3, 0.6245613098144531, 0.22553853690624237], [6, 0.7457385659217834, 0.7790691256523132], [5, 0.41731584072113037, 0.31907016038894653], [5, 0.4077262878417969, -0.7605134844779968], [5, -0.5758413672447205, 0.11818934231996536], [2, 0.3633080720901489, 0.34891554713249207], [5, 0.5574846267700195, -0.3271861970424652], [6, -0.7010883688926697, -0.7193737626075745], [1, -0.6358446478843689, 0.9205405712127686], [3, -0.6813554763793945, 0.05880691483616829]]
    inventory = [2365,1455,1800,2363,1517,1610,339]
    goal_pos = 		[20.713126053303558, 22.4917656457638]
    env_pos = 	[[71.55740708282099, 74.4406256301721], [40.81178027578002, 24.15972911540506], [20.713126053303558, 17.871765645763794]]
    env_pos = [np.array(l) for l in env_pos]
    env_pos = np.array(env_pos)
    tasks = ["CreateLevelPush"]

    # Define the data
    data = {
        "id": 0,
        "actions_taken": [
            [5, -0.36754459142684937, 0.507195770740509],
            [5, 0.6963381171226501, 0.3058029115200043],
            [6, -0.2735101878643036, -0.9138938188552856],
            [1, 0.5515308380126953, -0.8142970204353333],
            [1, 0.8675007820129395, 0.5794864892959595],
            [0, 0.7783870100975037, -0.15310797095298767],
            [3, -0.36247918009757996, 0.6076967120170593],
            [1, 0.983976423740387, -0.3070365786552429],
            [2, -0.4281010329723358, -0.8836000561714172],
            [0, 0.8419619798660278, -0.4214347004890442],
            [6, -0.3701498806476593, -0.5703181624412537],
            [5, 0.33143624663352966, -0.012288709171116352],
            [3, 0.1310325264930725, 0.09294066578149796],
            [3, 0.5017299652099609, 0.2953125536441803],
            [3, 0.13869720697402954, 0.606238842010498],
            [5, -0.29731348156929016, -0.2531295120716095],
            [4, 0.17405042052268982, -0.7616918683052063],
            [1, 0.48880717158317566, 0.5066838264465332],
            [5, -0.10166114568710327, -0.9610490202903748],
            [1, 0.5800946354866028, 0.8064578175544739],
            [4, 0.5425825715065002, -0.4308127462863922],
            [4, -0.6202167868614197, -0.3246026635169983],
            [5, 0.47436535358428955, 0.1674366295337677],
            [0, 0.07304829359054565, 0.3664814829826355],
            [5, 0.2890625298023224, 0.9470997452735901],
            [3, 0.4760686159133911, -0.22646929323673248],
            [6, -0.5337839722633362, -0.7236716747283936],
            [3, -0.6521638631820679, 0.35383471846580505],
            [3, -0.8308069109916687, -0.25732889771461487]
        ],
        "ball_is_basket": False,
        "ball_traces": "",
        "blocked_action_count": 0,
        "dense_reward_scale": 0.0,
        "different_walls": False,
        "done_marker_sec_goals": "",
        "done_target_sec_goals": "",
        "env_pos": [
            [72.95726723257057, 72.8525652480298],
            [48.09861175779888, 36.49049017073154],
            [11.73986395240389, 17.987485774328825]
        ],
        "episode_dense_reward": 0.0,
        "fps": 30.0,
        "goal_is_basket": False,
        "goal_pos": [11.73986395240389, 22.607485774328822],
        "has_reset": True,
        "init_dist": 0.3434188224990824,
        "invalid_action_count": 0,
        "inventory": [2365, 2272, 1583, 12, 1694, 1641, 132],
        "is_setup": True,
        "large_steps": 40,
        "line_traces": "",
        "marker_ball_traces": "",
        "marker_collided": False,
        "marker_line_traces": "",
        "marker_lines": "",
        "marker_must_hit": False,
        "marker_positions": "",
        "marker_sec_goals": "",
        "max_num_steps": 30,
        "moving_goal": False,
        "no_action_space_resample": False,
        "place_walls": False,
        "prev_dist": 0.7083237610411577,
        "scale": 12.19047619047619,
        "server_mode": False,
        "target_obj_start_pos": [48.09861175779888, 41.110490170731545],
        "target_positions": "",
        "target_sec_goals": "",
        "task": "CreateLevelPush-v0",
        "task_id": "CreateLevelPush-v0",
    }



    for task in tasks:
        done = False
        task_i = f'{task}-v0'
        env = gym.make(f'{task}-v0')
        settings = CreateGameSettings(
            evaluation_mode=True,
            max_num_steps=30,
            action_set_size=7,
            render_mega_res=False,
            render_ball_traces=False)
        trainer = TrainModel(DQN,
                             env)
        env.set_settings(settings)

        env.reset()

        env.set_task_id(task_i)
        count_outcome = 0
        max_iter = 0
        set_attributes_from_dict(env, data)
        st = 0
        ep_count  = 5
        dir = os.getcwd()
        print(env.env_pos)
        for action in data['actions_taken']:
            obs, reward, done, info = env.step(action)
            image_pil = Image.fromarray(obs)
            ep_dir = os.path.join(dir,'episodes_test')
            ep_dir = os.path.join(ep_dir,f"{ep_count}")
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)
            file_name = f'st_{st}.png'
            file_path = os.path.join(ep_dir, file_name)

            image_pil.save(file_path)
            st += 1
        print(info['cur_goal_hit'])







    #         print(f"{task_i} {info['cur_goal_hit']}")
    #         max_iter += 1
    #         if info['cur_goal_hit'] == 1:
    #             count_outcome = count_outcome + 1
    #             success_inventory = env.inventory
    #             env.reset()
    #             env.inventory = success_inventory
    #             concept_space_episode_id = trainer.concept_space.add_data('Episode')
    #             episode_id = concept_space_episode_id['elementId(n)'][0]
    #             trainer.concept_space.set_property(episode_id, 'Episode', 'succesfull_outcome', True)
    #             trainer.concept_space.set_property(episode_id, 'Episode', 'task', task_i, is_string=True)
    #             trainer.concept_space.set_property(episode_id, 'Episode', 'inventory',  ",".join(map(str, list(success_inventory))), is_string=True)
    #             print(actions_taken_in_episode)
    #             print(episode_id)
    #             trainer.concept_space.close()
    #             timestep = 0
    #
    #             encoded_inventory, objects_interacting_frames = trainer.action_embedder.get_inventory_embeddings(
    #                 env.inventory, object_extractor)
    #
    #             current_screen, encoded_state_t, state_id, action_tool_ids = trainer.get_current_state_graph(
    #                 obs, objects_interacting_frames,
    #                 episode_id, timestep
    #             )
    #             new_done = False
    #             for action in actions_taken_in_episode:
    #                 action_t = action[0]
    #                 pos1 = action[1]
    #                 pos2 = action[2]
    #                 if not new_done:
    #                     obs, reward, new_done, info = env.step(action)
    #                     plt.imshow(obs)
    #                     plt.show()
    #                     inventory_item_applied = env.inventory.item(action_t)
    #                     position_applied = [pos1, pos2]
    #
    #                     encoded_inventory, objects_interacting_frames = action_embedder.get_inventory_embeddings(
    #                     env.inventory, object_extractor)
    #
    #                     aff_id = trainer.wm.match_action_affordance(action_tool_ids, inventory_item_applied,
    #                                                             position_applied, reward,
    #                                                             timestep)
    #                     trainer.wm.concept_space.match_state_add_aff(state_id, aff_id)
    #
    #
    #                     current_screen, encoded_state_t_plus_1, state_id, action_tool_ids = trainer.get_current_state_graph(
    #                         obs, objects_interacting_frames,
    #                         episode_id, timestep + 1
    #                     )
    #
    #
    #                     affordance_effect = trainer.compute_effect(encoded_state_t, encoded_state_t_plus_1, reward)
    #                     encoded_state_t = encoded_state_t_plus_1
    #
    #                     timestep += 1
    #                     if aff_id is not None and state_id is not None:
    #                         trainer.wm.concept_space.match_state_add_aff_outcome(state_id, aff_id)
    #                         trainer.wm.concept_space.set_property(aff_id, 'Affordance', 'outcome', affordance_effect.tolist())
    #
    #
    #             # try:
    #             #     env.render('human')
    #             # except:
    #             #     pass
    #
    # with open('success_ep.pickle', 'wb') as handle:
    #     pickle.dump(succesfull_episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #
    #     # while not done:
    #     #     obs, reward, done, info = env.step(env.action_space.sample())
    #     #     env.render('human')
    #
    #     # frames, reward, hit_goal, dict_r = env.step([random.randint(0, 6)])
    #     # plt.imshow(frames)
    #     # plt.show()


if __name__ == '__main__':
    main()
