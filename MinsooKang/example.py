# code for explain


#visualizing in xml
import os
# visualizing in csv
#import pandas as pd

# Environment Setting
from flow.envs.ring.accel import AccelEnv
# Network Setting
from flow.networks.ring import RingNetwork
# Core Parameter
from flow.core.params import SumoParams  # Sumo Params
from flow.core.params import VehicleParams  # 차량 추가시 사용 거의 반필수
from flow.core.params import NetParams  # network parameter지정을 위한 parameter
from flow.core.params import InitialConfig  # 초기 차량의 구성 형태
from flow.core.params import TrafficLightParams  # 신호등 추가용
from flow.core.params import EnvParams  # 환경 parameter
from flow.core.experiment import Experiment  # experiment running Env

# User Model 사용시 변환의 여지가 있으나 차량이 어떤 주행 모델을 사용할 것인지

# 이건 Autonomous car의 driving model
from flow.controllers.car_following_models import IDMController
# 이건 Human-driven car의 driving model
from flow.controllers.routing_controllers import ContinuousRouter


# Network의 유형 선택, 이것을 import하면 이 부분 사용하는 것임
from flow.networks.ring import ADDITIONAL_NET_PARAMS
print(ADDITIONAL_NET_PARAMS)  # Network parameter 확인


# Project의 이름
name = "ring_example"

# Vechicle setting
vehicles = VehicleParams()
vehicles.add("human", acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=22)

# Network setting
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

# Initial Config 초기 차량 구성 설정
initial_config = InitialConfig()
initial_config = InitialConfig(spacing="uniform", perturbation=1)

# traffic light setting
traffic_lights = TrafficLightParams()

# Sumo param
sim_params = SumoParams(sim_step=0.1, render=True, emission_path='data')

# Environment param
env_params = EnvParams(additional_params=ADDITIONAL_NET_PARAMS)
# flow parameters 설정 frome above
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
net_params = NetParams(
    additional_params={
        'length': 230,
        'lanes': 2,
        'speed_limit': 30,
        'resolution': 40
    }
)

# number of time steps
flow_params['env'].horizon = 3000  # time step 수 지정
# sim step 0.1이면, 0.1*3000 -> 300s
exp = Experiment(flow_params)  # 여기서 flow parameter 지정한 것 실행

# run the sumo simulation
_ = exp.run(1, convert_to_csv=True)  # true시 csv 파일로 export됨
# exp.run(돌리는 횟수, csv로의 변환)
# convert to csv를 쓰지 않는 경우, import os 를 통해서 xml포맷으로 print가능
# csv파일인 경우, import pandas as pd를 통해서 pd.read_csv 함수 사용시 csv포맷 읽을 수 있음
emission_location = os.path.join(
    exp.env.sim_params.emission_path, exp.env.network.name)
print(emission_location | '-emission.xml')
# pd.read_csv(emission_location+'-emission.csv')
