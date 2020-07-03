from flow.envs.ring.accel import AccelEnv
from flow.networks.ring import RingNetwork  # flow.networks 사용시 environment추가
from flow.core.params import VehicleParams  # 차량 추가시 사용 거의 반필수
from flow.core.params import NetParams  # network parameter지정을 위한 parameter
from flow.core.params import InitialConfig  # 초기 차량의 구성 형태
from flow.core.params import TrafficLightParams  # 신호등 ㅊ추가용
# User Model 사용시 변환의 여지가 있으나 차량이 어떤 주행 모델을 사용할 것인지

# 이건 Autonomous car의 driving model
from flow.controllers.car_following_models import IDMController
# 이건 Human-driven car의 driving model
from flow.controllers.routing_controllers import ContinuousRouter


# Network의 유형 선택, 이것을 import하면 이 부분 사용하는 것임
from flow.networks.ring import ADDITIONAL_NET_PARAMS
print(ADDITIONAL_NET_PARAMS)  # Network parameter 확인
# Environment Setting


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
initial_config = InitialConfig(spacing="uniform", perturbation=1)

# traffic light setting
traffic_Lights = TrafficLightParams()
