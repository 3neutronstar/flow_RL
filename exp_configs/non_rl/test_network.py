# This file is just for looking at network without going cars.

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
# change only this
from Network.turnleft_network import Turnleft_Network, ADDITIONAL_NET_PARAMS
from flow.envs.base import Env
import gym
from abc import ABCMeta

net_params = NetParams(
    additional_params=ADDITIONAL_NET_PARAMS.copy()
)

env_params = EnvParams()

sim_params = SumoParams(render=True)

initial_config = InitialConfig()

vehicles = VehicleParams()
vehicles.add('human', num_vehicles=1,)

flow_params = dict(
    exp_tag='test_network',
    env_name=TestEnv,
    network=Turnleft_Network,  # change only this
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)
