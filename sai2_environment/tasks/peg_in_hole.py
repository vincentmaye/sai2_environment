from sai2_environment.tasks.task import Task
import numpy as np

#ToDo : CHANGE EVERYTHING FOR PEG IN HOLE

class PegInHole(Task):
    def __init__(self, task_name, redis_client, simulation=True):
        self._task_name = task_name
        self._client = redis_client
        self._simulation = simulation
        self.TARGET_OBJ_POSITION_KEY  = "sai2::ReinforcementLearning::peg_in_hole::object_position" # Changed this to peg in hole
        self.GOAL_POSITION_KEY  = "sai2::ReinforcementLearning::peg_in_hole::goal_position" # Changed this to peg in hole

        if simulation:
            
            self.goal_position = self._client.redis2array(self._client.get(self.GOAL_POSITION_KEY))
            self.current_obj_position = self.get_current_position()
            self.last_obj_position = self.current_obj_position
            self.total_distance = self.euclidean_distance(self.goal_position, self.current_obj_position)
        else:
            #setup the things that we need in the real world
            self.goal_position = None
            self.current_obj_position = None
            self.last_obj_position = None
            self.total_distance = None

    def compute_reward(self):
        if self._simulation:
            self.last_obj_position = self.current_obj_position
            self.current_obj_position = self.get_current_position()
            d0 = self.euclidean_distance(self.goal_position, self.last_obj_position)
            d1 = self.euclidean_distance(self.goal_position, self.current_obj_position)

            reward = (d0 - d1)/self.total_distance
            #radius of target location is 0.04
            done = np.linalg.norm(self.goal_position - self.current_obj_position) < 0.04
        else:
            #TODO
            reward = 0
            done = False

        return reward, done

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def get_current_position(self):
        return self._client.redis2array(self._client.get(self.TARGET_OBJ_POSITION_KEY))

