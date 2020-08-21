# This file is just for looking at network

from flow.envs import TestEnv
# the Experiment class is used for running simulations
from flow.core.experiment import Experiment

# all other imports are standard
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams
# map data
from Network.custom_network import Custom_Network
from Network.custom_network import net_params
from flow.envs.base import Env
import gym
from abc import ABCMeta

env_params = EnvParams()
sim_params = SumoParams(render=True)
initial_config = InitialConfig()
vehicles = VehicleParams()
vehicles.add('human', num_vehicles=1,)

flow_params = dict(
    exp_tag='custom_network',
    env_name=TestEnv,
    network=Custom_Network,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)
