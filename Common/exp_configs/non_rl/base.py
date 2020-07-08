from flow.networks import Network
from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.core.experiment import Experiment
# contributor: Gihong Lee, Minsoo Kang


class myNetwork(Network):
    pass


ADDITIONAL_NET_PARAMS = {
    # "st_line": 40,
    # "num_lanes": 1,
    # "speed_limit": 30,
}

vehicles = VehicleParams()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=22)


flow_params = dict(
    # name of the experiment
    exp_tag='ring',

    # name of the flow environment the experiment is running on
    env_name=AccelEnv,

    # name of the network class the experiment is running on
    network=myNetwork,  # you've already declared first

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.1,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=1500,
        additional_params=ADDITIONAL_ENV_PARAMS,
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
    initial=InitialConfig(
        bunching=20,
    ),
)


class myNetwork(myNetwork):
    def specify_nodes(self, net_params):
        # one of the elements net_params will need is a "radius" value
        r = net_params.additional_params["st_line"]

        # specify the name and position (x,y) of each node
        nodes = [{"id": "LU", "x": -r,  "y": +r},
                 # ...
                 ]
        return nodes

    def specify_edges(self, net_params):
        r = net_params.additional_params["st_line"]
        edgelen = r
        # this will let us control the number of lanes in the network
        lanes = net_params.additional_params["num_lanes"]
        # speed limit of vehicles in the network
        speed_limit = net_params.additional_params["speed_limit"]
        # L: left, R: right, U: Up D:Down
        edges = [
            {
                "id": "edge0",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LU",
                "to": "CL",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(-pi/2, 0, 40)]
            },
            # ...
        ]

    def specify_routes(self, net_params):
        rts = {"edge0": [(["edge0", "edge1", "edge2", "edge3", "edge4", "edge5", "edge6", "edge7"], 0.5), (["edge0", "edge1", "edge2", "edge18", "edge22", "edge7"], 0.5)],
               # ...
               }
        return rts


# number of time steps
flow_params['env'].horizon = 1500
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1)
