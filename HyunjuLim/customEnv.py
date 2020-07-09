# tutorial_08: Creating Custom Environments

# we will create an environment in which the accelerations of a handful of vehicles in the network 
# are specified by a single centralized agent, with the objective of the agent being to improve 
# the average speed of all vehicle in the network.

# import the base environment class
from flow.envs import Env

# define the environment class, and inherit properties from the base environment class
class myEnv(Env):
    pass

"""
# Env 상속 받아서 건들 수 있는 것
action_space
observation
apply_rl_actions
get_state
compute_reward
"""

# ADDITIONAL_ENV_PARAMS 
# storing information on the maximum possible accelerations and decelerations by the autonomous vehicles in the network. 
ADDITIONAL_ENV_PARAMS = {
    "max_accel": 1,
    "max_decel": 1,
}

# action_space
# The action_space method defines the number and bounds of the actions provided by the RL agent.
# In order to define these bounds with an OpenAI gym setting, we use several objects located within gym.spaces.
# For instance, the Box object is used to define a bounded array of values in R.
from gym.spaces.box import Box
from gym.spaces import Tuple

# Once we have imported the above objects, we are ready to define the bounds of our action space.
class myEnv(myEnv):

    @property
    def action_space(self):
        num_actions = self.initial_vehicles.num_rl_vehicles
        accel_ub = self.env_params.additional_params["max_accel"]
        accel_lb = - abs(self.env_params.additional_params["max_decel"])

        return Box(low=accel_lb,
                   high=accel_ub,
                   shape=(num_actions,))

# observation_space¶
class myEnv(myEnv):  # update my environment class

    @property
    def observation_space(self):
        return Box(
            low=0,
            high=float("inf"),
            shape=(2*self.initial_vehicles.num_vehicles,),
        )

# apply_rl_actions
class myEnv(myEnv):  # update my environment class

    def _apply_rl_actions(self, rl_actions):
        # the names of all autonomous (RL) vehicles in the network
        rl_ids = self.k.vehicle.get_rl_ids()

        # use the base environment method to convert actions into accelerations for the rl vehicles
        self.k.vehicle.apply_acceleration(rl_ids, rl_actions)

# get_state
import numpy as np

class myEnv(myEnv):  # update my environment class

    def get_state(self, **kwargs):
        # the get_ids() method is used to get the names of all vehicles in the network
        ids = self.k.vehicle.get_ids()

        # we use the get_absolute_position method to get the positions of all vehicles
        pos = [self.k.vehicle.get_x_by_id(veh_id) for veh_id in ids]

        # we use the get_speed method to get the velocities of all vehicles
        vel = [self.k.vehicle.get_speed(veh_id) for veh_id in ids]

        # the speeds and positions are concatenated to produce the state
        return np.concatenate((pos, vel))


# compute_reward
import numpy as np

class myEnv(myEnv):  # update my environment class

    def compute_reward(self, rl_actions, **kwargs):
        # the get_ids() method is used to get the names of all vehicles in the network
        ids = self.k.vehicle.get_ids()

        # we next get a list of the speeds of all vehicles in the network
        speeds = self.k.vehicle.get_speed(ids)

        # finally, we return the average of all these speeds as the reward
        return np.mean(speeds)


