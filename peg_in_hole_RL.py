import redis
from sai2_environment.robot_env import RobotEnv
from sai2_environment.action_space import ActionSpace

import numpy as np
import time
from PIL import Image
import torch

from sai2_environment.reinforcement_learning.utils.run_utils import setup_logger_kwargs
from sai2_environment.reinforcement_learning.rl_algos import sac

from subprocess import Popen

# Start the environment and controller
"""
Popen("killall terminator && killall controller_peg_exe", shell=True)

Popen("terminator -e ./sim_peg_exe", shell=True, cwd='/home/msrm-student/sai2/apps/RobotLearningApp/bin/02-peg_in_hole')
time.sleep(2)
Popen("terminator --new-tab -e ./controller_peg_exe", shell=True,  cwd='/home/msrm-student/sai2/apps/RobotLearningApp/bin/02-peg_in_hole')
time.sleep(2)
"""
def main():
    # If Debug mode don't log
    debug = True
    #*** Stuff from OpenAI ***#
    logger_kwargs = setup_logger_kwargs("peg_in_hole_test", 0, datestamp=True, data_dir='sai2_environment/reinforcement_learning/logs/') # Vars: exp_name, seed
    torch.set_num_threads(torch.get_num_threads())

    # Robot stuff
    action_space = ActionSpace.DELTA_EE_POSE_IMPEDANCE 
    #action_space = ActionSpace.ABS_JOINT_POSITION_IMPEDANCE
    blocking_action = True
    env = RobotEnv(name='peg_in_hole',
                   simulation=True,
                   action_space=action_space,
                   isotropic_gains=True,
                   render=False,
                   blocking_action=blocking_action,
                   rotation_axis=(0, 0, 1))    
    # Run SAC
    sac(env, logger_kwargs = logger_kwargs, DEBUG=debug)
    """ env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1 """


if __name__ == "__main__":
    main()