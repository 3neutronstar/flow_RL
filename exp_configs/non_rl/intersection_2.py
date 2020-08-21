
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.networks import Network

from Network.intersection_network import IntersectionNetwork
from controllers.intersection_lane_controller import intersection_lane_controller

vehicles = VehicleParams()
vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             lane_change_controller=num_vehicles=18)

sim_params = SumoParams(sim_step=0.1, render=True)

initial_config = InitialConfig(spacing="uniform",
                               bunching=40,
                               )

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS,
                       )

additional_net_params = {
    "length": 40,
    "lanes": 1,
    "speed_limit": 30,
    "resolution": 40,
}
net_params = NetParams(additional_params=additional_net_params)

flow_params = dict(
    exp_tag='intersection',
    env_name=AccelEnv,
    network=IntersectionNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)
