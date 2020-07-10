from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs import WaveAttenuationPOEnv
from Network.custom_network import Custom_Network


vehicles = VehicleParams()
vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=14)

sim_params = SumoParams(sim_step=0.1, render=True)

initial_config = InitialConfig(spacing="uniform", bunching=40)

ADDITIONAL_ENV_PARAMS = {

}

env_params = EnvParams(horizon=1500,
                       additional_params=ADDITIONAL_ENV_PARAMS
                       )

ADDITIONAL_NET_PARAMS = {

}
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
net_params = NetParams(additional_params=additional_net_params)


flow_params = dict(
    # name of the experiment
    exp_tag='custom_network',
    # name of the flow environment the experiment is running on
    env_name=WaveAttenuationPOEnv,
    # name of the network class the experiment is running on
    network=Custom_Network,  # RingNetwork_custom
    # simulator that is used by the experiment
    simulator='traci',
    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=sim_params,
    # environment related parameters (see flow.core.params.EnvParams)
    env=env_params,
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
    initial=initial_config
)
