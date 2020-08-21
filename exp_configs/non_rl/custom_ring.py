from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.networks import Network
import numpy as np
from numpy import pi, sin, cos, linspace
from Network.custom_ring import RingNetwork_custom, ADDITIONAL_NET_PARAMS

vehicles = VehicleParams()


vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=14)
sim_params = SumoParams(sim_step=0.1, render=True)

initial_config = InitialConfig(spacing="uniform", bunching=40)

env_params = EnvParams(
    horizon=1500,
    additional_params=ADDITIONAL_ENV_PARAMS)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()


net_params = NetParams(additional_params=additional_net_params)


flow_params = dict(
    # name of the experiment
    exp_tag='custom_ring',
    # name of the flow environment the experiment is running on
    env_name=AccelEnv,
    # name of the network class the experiment is running on
    network=RingNetwork_custom,  # RingNetwork_custom
    # simulator that is used by the experiment
    simulator='traci',
    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.1,
    ),
    # environment related parameters (see flow.core.params.EnvParams)
    env=env_params,
    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=additional_net_params,
    ),
    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,
    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        bunching=20,
    ),
)
