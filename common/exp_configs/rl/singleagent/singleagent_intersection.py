from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.envs import WaveAttenuationPOEnv, AccelEnv
from Network.intersection_network import IntersectionNetwork
from Network.intersection_network import ADDITIONAL_NET_PARAMS
# time horizon of a single rollout
HORIZON = 1000
# number of rollouts per training iteration
N_ROLLOUTS = 8
# number of parallel workers
N_CPUS = 2

# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=0
    ),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=16)

vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    initial_speed=25,
    num_vehicles=1)


flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_the_intersection",

    # name of the flow environment the experiment is running on
    env_name=AccelEnv,

    # name of the network class the experiment is running on
    network=IntersectionNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=False
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=1500,  # after training, play time
        clip_actions=False,
        additional_params={
            "max_accel": 3,
            "max_decel": 1,
            "target_velocity": 30,
            "sort_vehicles": False
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)