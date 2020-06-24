import redis
from sai2_environment.robot_env import RobotEnv
from sai2_environment.action_space import ActionSpace

import numpy as np
import time
from PIL import Image


def main():

    action_space = ActionSpace.DELTA_EE_POSE_IMPEDANCE
    blocking_action = True

    env = RobotEnv(name='peg_in_hole',
                   simulation=True,
                   action_space=action_space,
                   isotropic_gains=True,
                   render=True,
                   blocking_action=blocking_action,
                   rotation_axis=(0, 0, 1))    

    episodes = 50
    steps = 200

    start_time = time.time()    

    for episode in range(episodes):
        
        print("Episode: {}; Elapsed Time: {} minutes".format(episode, round((time.time()-start_time)/60), 4))
        obs = env.reset()
        for step in range(steps):
            position = np.around(np.random.uniform(low=-0.1, high=0.1, size=(3,)), 3)
            rotation = np.around(np.random.uniform(low=-0.1, high=0.1, size=(1,)), 2)
            stiffness_linear = np.around(np.random.uniform(low=-50, high=50, size=(1,)),2)
            stiffness_rot = np.around(np.random.uniform(low=-2, high=2, size=(1,)), 2)
            action = np.concatenate((position,rotation,stiffness_linear,stiffness_rot))


            if not blocking_action:
                time.sleep(0.1)
            obs, reward, done, info = env.step(action)

if __name__ == "__main__":
    main()
