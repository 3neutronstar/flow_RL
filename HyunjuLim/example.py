# tutorial_01 setting up an Environment
# we will not be training any autonomous agents

# AccelEnv는 fully observable network에서 a static number of vehicles를 train 
from flow.envs.ring.accel import AccelEnv

# SumoParams specifies simulation-specific variables.
from flow.core.params import SumoParams
sim_params = SumoParams(sim_step=0.1, render=True, emission_path='data')

# EnvParams specify environment and experiment-specific parameters 
# that either affect the training process or the dynamics of various components 
# within the network
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
print(ADDITIONAL_ENV_PARAMS)

# AUTERT_ENV_PARAMS 변수를 가져오면 환경과 관련된 리워드 함수를 계산할 때 사용되는 
# "target_velocity"라는 하나의 엔트리로만 구성되는 것을 알 수 있다. 
# EnvParams 객체를 생성할 때 사용한다.
from flow.core.params import EnvParams
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
