import pandas as pd
import os
from flow.core.experiment import Experiment
from flow.core.params import EnvParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.core.params import SumoParams
from flow.envs.ring.accel import AccelEnv
from flow.core.params import TrafficLightParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import IDMController
from flow.core.params import VehicleParams
from flow.networks.ring import RingNetwork
name = "ring_example"

vehicles = VehicleParams()

vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=36)

print(ADDITIONAL_NET_PARAMS)


#net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

initial_config = InitialConfig(spacing="uniform", perturbation=1)

traffic_lights = TrafficLightParams()

sim_params = SumoParams(sim_step=0.1, render=True, emission_path='data')
# render true는 gui 켜는 것
print(ADDITIONAL_ENV_PARAMS)
net_params = NetParams(
    additional_params={
        'length': 230,
        'lanes': 2,
        'speed_limit': 30,
        'resolution': 40
    }
)
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
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

emission_location = os.path.join(
    exp.env.sim_params.emission_path, exp.env.network.name)
print(emission_location + '-emission.xml')

pd.read_csv(emission_location + '-emission.csv')
