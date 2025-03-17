import random
import os
import gym
import numpy as np
from PIL import Image
from gym.envs.registration import register
from affordance_learning.action_observation.args import get_args
from affordance_learning.action_observation.env_interface import register_env_interface
from affordance_learning.action_observation.create_env_interface import CreateGameInterface, CreatePlayInterface
import matplotlib.pyplot as plt

path_ds_aff = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "create_aff_ds\\objects_obs_grouped")
register_env_interface('^Create((?!Play).)*$', CreateGameInterface)
register_env_interface('^Create(.*?)Play(.*)?$', CreatePlayInterface)
register_env_interface('^StateCreate(.*?)Play(.*)?$', CreatePlayInterface)

register(
    id='CreateGamePlay-v0',
    entry_point='affordance_learning.action_observation.create_play:CreatePlay',
)


def save_ndarray_as_image(ndarray, output_path):
    """Saves a NumPy ndarray as an image using Matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set the face color of the figure and the axes
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.imshow(ndarray)
    ax.axis('off')  # Turn off the axis

    # Save the figure with the specified face color
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.show()
    plt.close(fig)
    print(f"Image saved at {output_path}")


def save_ndarray_as_image_pil(ndarray, output_path):
    """Transforms a NumPy ndarray to an RGB image and saves it."""
    # Ensure the ndarray is in the correct format (uint8 and 3 channels)
    if ndarray.dtype != np.uint8:
        ndarray = (255 * (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min())).astype(np.uint8)

    if ndarray.shape[2] == 3:  # Check if the ndarray has 3 channels
        image = Image.fromarray(ndarray, 'RGB')
    else:
        raise ValueError("The ndarray does not have 3 channels.")

    # Save the image using PIL
    image.save(output_path)
    print(f"Image saved at {output_path}")



def main():
    env = gym.make(f'CreateGamePlay-v0')

    # settings = CreateGameSettings(
    #     evaluation_mode=False,
    #     max_num_steps=60,
    #     render_mega_res=True,
    #     render_ball_traces=False)
    args = get_args()
    env.update_args(args)
    # env.set_settings(settings)

    #env.reset()
    done = False
    frames = []
    allowed_actions = env.ALL_TOOLS
    #allowed_actions.sort()
    tool_folder = path_ds_aff

    for action in allowed_actions:
        # env.jf['env'][0]['pos'] = [0.5, 0.75]
        # env.jf['rnd']['marker_ball:0'] = '[0.5,0.75]'
        tool_type = action.tool_type
        i = 0
        done = False
        print(action)
        tool_folder = os.path.join(path_ds_aff, tool_type, f"{action.tool_id}")
        if not os.path.exists(tool_folder):
            os.makedirs(tool_folder)

        obs, reward, done, collision = env.step(action.tool_id)

        # res = env.render()
        # image_obs = transforms.ToPILImage()(obs)
        # plt.imshow(image_obs)
        # plt.show()
        j = 0
        if collision:
            for img in list(obs):
                save_ndarray_as_image_pil(img, f'{tool_folder}\\{j}.png')
                j+=1
        else:
            print(collision)
            # im = Image.fromarray(np.uint8(img))
            # im.save(f'{tool_folder}\\{i}.png')

            # if i == 0:
            #     #no_op_action = [np.where(env.inventory == action)[0][0], 0.5, 0.3]
            #     no_op_action = [action.tool_id, 0.5, 0.3]
            #
            #     obs, reward, done = env.step(action.tool_id)
            #     #res = env.render()
            #     image_obs = transforms.ToPILImage()(obs)
            #     plt.imshow(image_obs)
            #     plt.show()
            #     im = Image.fromarray(image_obs)
            #     im.save(f'{tool_folder}\\{i}.png')
            # else:
            #     #no_op_action = [np.where(env.inventory == 2365)[0][0], 1.0, 1.0]
            #     no_op_action = [2365, 1.0, 1.0]
            #     obs, reward, done, info = env.step(2365)
            #     #res = env.render()
            #     image_obs = transforms.ToPILImage()(obs)
            #     im = Image.fromarray(image_obs)
            #     im.save(f'{tool_folder}\\{i}.png')
            # if np.any(env.inventory == action):
            #     print(action)
            #     tool_folder = os.path.join(path_ds_aff, f"{action}")
            #     if not os.path.exists(tool_folder):
            #         os.makedirs(tool_folder)
            #
            #     if i == 0:
            #         no_op_action = [np.where(env.inventory == action)[0][0], 0.5, 0.3]
            #
            #         obs, reward, done, info = env.step(no_op_action)
            #         res = env.render()
            #         image_obs = transforms.ToPILImage()(obs)
            #         plt.imshow(image_obs)
            #         plt.show()
            #         im = Image.fromarray(res)
            #         im.save(f'{tool_folder}\\{i}.png')
            #     else:
            #         no_op_action = [np.where(env.inventory == 2365)[0][0], 1.0, 1.0]
            #         obs, reward, done, info = env.step(no_op_action)
            #         res = env.render()
            #         image_obs = transforms.ToPILImage()(obs)
            #         im = Image.fromarray(res)
            #         im.save(f'{tool_folder}\\{i}.png')
            i += 1

        env.reset()

    env.close()

if __name__ == "__main__":
    main()
