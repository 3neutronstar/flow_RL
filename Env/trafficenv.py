# import the base environment class
from flow.envs import Env
from gym.spaces.box import Box
from gym.spaces import Tuple
import numpy as np
# define the environment class, and inherit properties from the base environment class


class myEnv(Env):
    pass


ADDITIONAL_ENV_PARAMS = {
    "max_accel": 1,
    "max_decel": 1,
}


class myEnv(myEnv):

    @property
    def action_space(self):
        num_actions = self.initial_vehicles.num_rl_vehicles
        accel_ub = self.env_params.additional_params["max_accel"]
        accel_lb = - abs(self.env_params.additional_params["max_decel"])

        return Box(low=accel_lb,
                   high=accel_ub,
                   shape=(num_actions,))

    def observation_space(self):
        return Box(
            low=0,
            high=float("inf"),
            shape=(2*self.initial_vehicles.num_vehicles,),
        )

    def _apply_rl_actions(self, rl_actions):
        # the names of all autonomous (RL) vehicles in the network
        rl_ids = self.k.vehicle.get_rl_ids()

        # use the base environment method to convert actions into accelerations for the rl vehicles
        self.k.vehicle.apply_acceleration(rl_ids, rl_actions)
        # apply_acceleration (list of str, list of float) -> None: converts an action, or a list of actions, into accelerations to the specified vehicles (in simulation)
        # apply_lane_change (list of str, list of {-1, 0, 1}) -> None: converts an action, or a list of actions, into lane change directions for the specified vehicles (in simulation)
        # choose_route (list of str, list of list of str) -> None: converts an action, or a list of actions, into rerouting commands for the specified vehicles (in simulation)

    def get_state(self, **kwargs):
        # the get_ids() method is used to get the names of all vehicles in the network
        ids = self.k.vehicle.get_ids()

        # we use the get_absolute_position method to get the positions of all vehicles
        pos = [self.k.vehicle.get_x_by_id(veh_id) for veh_id in ids]

        # we use the get_speed method to get the velocities of all vehicles
        vel = [self.k.vehicle.get_speed(veh_id) for veh_id in ids]

        # the speeds and positions are concatenated to produce the state
        return np.concatenate((pos, vel))

    def compute_reward(self, rl_actions, **kwargs):
        # the get_ids() method is used to get the names of all vehicles in the network
        ids = self.k.vehicle.get_ids()

        # we next get a list of the speeds of all vehicles in the network
        speeds = self.k.vehicle.get_speed(ids)

        # finally, we return the average of all these speeds as the reward
        return np.mean(speeds)
