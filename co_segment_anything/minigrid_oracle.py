import gym
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper
import numpy as np

# -- Define a very simple oracle agent --
def simple_oracle(obs):
    """
    Oracle: if front is clear, go forward (action=2), else turn right (action=1).
    Works best with partial RGB observation.
    """
    image = obs['image']
    front_cell = image[0][1]  # Front view in agent's perspective
    obj_type = front_cell[0]  # 0 means empty
    if obj_type == 0:
        return 2  # forward
    else:
        return 1  # turn right

def oracle_to_adjacent_goal(obs):
    image = obs['image']
    agent_dir = obs['direction']

    agent_pos = None
    goal_pos = None

    # Find agent and goal positions
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cell_type = image[i][j][0]
            cell_color = image[i][j][1]
            if cell_type == 10 and cell_color == 2:  # agent
                agent_pos = (i, j)
            elif cell_type == 8:  # goal
                goal_pos = (i, j)

    if agent_pos is None or goal_pos is None:
        return 1  # fallback: turn right

    # Determine relative position
    delta = (goal_pos[0] - agent_pos[0], goal_pos[1] - agent_pos[1])

    # If goal is not adjacent, just turn for now
    if delta not in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        return 1  # turn right until we find it

    # Determine desired direction to face goal
    dir_map = {
        (0, 1): 0,   # right
        (1, 0): 1,   # down
        (0, -1): 2,  # left
        (-1, 0): 3   # up
    }

    desired_dir = dir_map[delta]

    if agent_dir == desired_dir:
        return 2  # move forward
    else:
        return 1  # rotate right


def wall_aware_oracle(obs):
    image = obs['image']
    agent_dir = obs['direction']
    agent_pos = None
    goal_pos = None

    # Find agent and goal in the grid
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            obj_type = image[row][col][0]
            if obj_type == 10:
                agent_pos = (row, col)
            elif obj_type == 8:
                goal_pos = (row, col)

    if agent_pos is None or goal_pos is None:
        print("⚠️ Agent or goal not found")
        return 1  # turn right to stall

    ay, ax = agent_pos
    gy, gx = goal_pos
    dx = gx - ax
    dy = gy - ay

    # Decide direction to move toward goal
    if abs(dy) > 0:
        desired_dir = 1 if dy > 0 else 3  # down or up
    elif abs(dx) > 0:
        desired_dir = 0 if dx > 0 else 2  # right or left
    else:
        print("✅ Reached goal!")
        return 6

    # Step 1: rotate to desired direction
    delta = (desired_dir - agent_dir) % 4
    if delta == 1:
        return 1  # right
    elif delta == 3:
        return 0  # left
    elif delta == 2:
        return 1  # 180° turn

    # Step 2: we're now aligned — check if wall in front
    DIR_TO_VEC = {
        0: (1, 0),  # right
        1: (0, 1),  # down
        2: (-1, 0),  # left
        3: (0, -1),  # up
    }

    dy, dx = DIR_TO_VEC[agent_dir]
    fy = agent_pos[0] + dy
    fx = agent_pos[1] + dx

    # dy_f, dx_f = DIR_TO_VEC[agent_dir]
    # fy, fx = ay + dy_f, ax + dx_f

    if 0 <= fy < image.shape[0] and 0 <= fx < image.shape[1]:
        obj_type_in_front = image[fy][fx][0]
        print(f"👀 Front cell at {(fy, fx)} has object type {obj_type_in_front}")
        if obj_type_in_front == 2:
            print("🚧 Wall in front — turning")
            return 1
        else:
            return 2  # floor or goal — move forward
    else:
        print("🚧 Front cell out of bounds — turning")
        return 1


class SmarterOracle:
    def __init__(self):
        self.blocked_dirs = set()

    def get_action(self, obs):
        image = obs['image']
        agent_dir = obs['direction']
        agent_pos = None
        goal_pos = None

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j][0] == 10:
                    agent_pos = (i, j)
                elif image[i][j][0] == 8:
                    goal_pos = (i, j)

        if agent_pos is None or goal_pos is None:
            return 1  # rotate

        ay, ax = agent_pos
        gy, gx = goal_pos
        dx = gx - ax
        dy = gy - ay

        # Prefer vertical first
        preferred_dirs = []
        if dy != 0:
            preferred_dirs.append(1 if dy > 0 else 3)
        elif dx != 0:
            preferred_dirs.append(0 if dx > 0 else 2)

        # Filter out blocked directions
        preferred_dirs = [d for d in preferred_dirs if d not in self.blocked_dirs]

        # If all preferred are blocked, try any other direction
        if not preferred_dirs:
            preferred_dirs = [d for d in range(4) if d not in self.blocked_dirs]

        if not preferred_dirs:
            print("🚧 All directions blocked — reset memory")
            self.blocked_dirs.clear()
            return 1

        desired_dir = preferred_dirs[0]

        # Rotate toward desired_dir
        delta = (desired_dir - agent_dir) % 4
        if delta == 1:
            return 1
        elif delta == 3:
            return 0
        elif delta == 2:
            return 1

        # Facing correct direction — check what's in front

        DIR_TO_VEC = {
            0: (1, 0),  # right
            1: (0, 1),  # down
            2: (-1, 0),  # left
            3: (0, -1),  # up
        }

        dy, dx = DIR_TO_VEC[agent_dir]
        fy = agent_pos[0] + dy
        fx = agent_pos[1] + dx

        if 0 <= fy < image.shape[0] and 0 <= fx < image.shape[1]:
            front_obj = image[fy][fx][0]
            if front_obj == 8:  # goal
                print("🎯 Goal is directly in front — moving forward!")
                return 2
            if front_obj == 2:  # wall
                print(f"🚧 Blocked at {fy, fx}, adding dir {agent_dir} to memory")
                self.blocked_dirs.add(agent_dir)
                return 1  # rotate
            else:
                self.blocked_dirs.clear()
                return 2  # move forward
        else:
            print("🚧 Front cell out of bounds")
            self.blocked_dirs.add(agent_dir)
            return 1



# -- Create and wrap environment --
def run_oracle():
    import random
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")

    env = FullyObsWrapper(env)
    seed = random.randint(0, 10000)
    obs, _ = env.reset(seed=seed)

    mem_oracle = SmarterOracle()

    env.unwrapped.agent_pos = (random.randint(1, 3), random.randint(1, 3))
    env.unwrapped.agent_dir = random.randint(0, 3)

    for step in range(12):
        # Choose oracle action
        action = mem_oracle.get_action(obs)

        # Take the action
        obs, reward, done, _, _ = env.step(action)

        # Render frame after action
        frame = env.render()

        # Plot the frame
        plt.imshow(frame)
        plt.title(f"Step {step + 1} | Action: {action} | Reward: {reward}")
        plt.axis('off')
        plt.show()

        if done:
            print("Episode finished")
            break

    env.close()
