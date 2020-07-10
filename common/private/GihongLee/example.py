import flow
from flow.networks.ring import RingNetwork
#This network, as well as all other networks in Flow,
#  is parametrized by the following arguments:
# name
# vehicles
# net_params
# initial_config
# traffic_lights

name = "ring_example" #no effect on the ytpe of network

# vehicleparams
from flow.core.params import VehicleParams
vehicles = VehicleParams() 
# add함수에 관련 https://flow.readthedocs.io/en/latest/flow.core.html?highlight=vehicleparam#flow.core.params.VehicleParams
# 참고

# vehicle's routing behavior 
#  the acceleration behavior of all vehicles will be defined by the Intelligent Driver Model (IDM)
from flow.controllers.car_following_models import IDMController
# ContinuousRouter controller is used to perpetually reroute all vehicles to the initial set route.
from flow.controllers.routing_controllers import ContinuousRouter

vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=22)


#NETPARAMS = network-specific parameters used to define the shape and properties of a network.
from flow.networks.ring import ADDITIONAL_NET_PARAMS

print(ADDITIONAL_NET_PARAMS)
# equired parameters are:

# length: length of the ring road
# lanes: number of lanes
# speed: speed limit for all edges
# resolution: resolution of the curves on the ring. Setting this value to 1 converts the ring to a diamond.

from flow.core.params import NetParams

net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

#INITIALCONFIG=> specifies parameters that affect the positioning of vehicle in the network at the start of a simulation.
from flow.core.params import InitialConfig
initial_config = InitialConfig(spacing="uniform", perturbation=1)

#TRAFFICLIGHTPARAMS=> positions and types of traffic lights in the network
from flow.core.params import TrafficLightParams
traffic_lights = TrafficLightParams()

#SETTING UP AN ENVIRONMNET
from flow.envs.ring.accel import AccelEnv
# Envrionments in Flow are parametrized by several components,
# sim_params
# env_params
# network
# net_params
# initial_config
# network
# simulator
#  sim_params, env_params, and network are the primary parameters of an environment
# Sumo envrionments in Flow are parametrized by three components:
# SumoParams
# EnvParams
# Network

#  useful parameter is emission_path, which is used to specify the path where the emissions output will be generated.
# SUMOPARAMS
from flow.core.params import SumoParams
sim_params = SumoParams(sim_step=0.1, render=True, emission_path='data')

# ENVPARAMS=> affect the training process or the dynamics of various components within the network
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
print(ADDITIONAL_ENV_PARAMS)

#Importing the ADDITIONAL_ENV_PARAMS variable, 
# we see that it consists of only one entry, "target_velocity", which is used when computing the reward function associated with the environment
from flow.core.params import EnvParams
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

#SETTING UP AND RUNNING THE EXPERIMENT
from flow.core.experiment import Experiment

flow_params = dict(
    exp_tag='ring_example',
    env_name=AccelEnv,
    network=RingNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
    tls=traffic_lights,
)

# number of time steps
flow_params['env'].horizon = 3000
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1, convert_to_csv=True)

import os

emission_location = os.path.join(exp.env.sim_params.emission_path, exp.env.network.name)
print(emission_location + '-emission.xml')

# Running RLlib Experiments

# Setting up Network Parameters
# name
# vehicles
# net_params
# initial_config

import flow.networks as networks
print(networks.__all__)

from flow.networks import RingNetwork
# ring road network class
network_name = RingNetwork
# input parameter classes to the network class
from flow.core.params import NetParams, InitialConfig

# name of the network
name = "training_example"

# network-specific parameters
from flow.networks.ring import ADDITIONAL_NET_PARAMS
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

# initial configuration to vehicles
initial_config = InitialConfig(spacing="uniform", perturbation=1)

### test