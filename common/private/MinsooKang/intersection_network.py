# import Flow's base network class
from numpy import pi, sin, cos, linspace
from flow.networks import Network
from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS

from flow.core.experiment import Experiment

# define the network class, and inherit properties from the base network class


class myNetwork(Network):
    pass


ADDITIONAL_NET_PARAMS = {
    "st_line": 40,
    "num_lanes": 1,
    "speed_limit": 30,
}


class myNetwork(myNetwork):  # update my network class

    def specify_nodes(self, net_params):
        # one of the elements net_params will need is a "radius" value
        r = net_params.additional_params["st_line"]

        # specify the name and position (x,y) of each node
        nodes = [{"id": "LU", "x": -r,  "y": +r},  # 1
                 {"id": "RU",  "x": +r,  "y": +r},  # 2
                 {"id": "LD",    "x": -r,  "y": -r},  # 3
                 {"id": "RD",   "x": +r, "y": -r},  # 4
                 {"id": "CL",   "x": -r, "y": 0},  # 5
                 {"id": "CR",   "x": +r, "y": 0},  # 6
                 {"id": "CU",   "x": 0, "y": r},  # 7
                 {"id": "CD",   "x": 0, "y": -r}]  # 8

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
                "to": "LD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(-pi/2, 0, 40)]
            },
            {
                "id": "edge1",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LD",
                "to": "RD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(0, pi/2, 40)]
            },
            {
                "id": "edge2",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RD",
                "to": "RU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi/2, pi, 40)]
            },
            {
                "id": "edge3",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RU",
                "to": "LU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge4",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LD",
                "to": "LU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(-pi/2, 0, 40)]
            },
            {
                "id": "edge5",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RD",
                "to": "LD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(0, pi/2, 40)]
            },
            {
                "id": "edge6",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RU",
                "to": "RD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi/2, pi, 40)]
            },
            {
                "id": "edge7",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LU",
                "to": "RU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            # {
            #     "id": "edge8",
            #     "numLanes": lanes,
            #     "speed": speed_limit,
            #     "from": "CL",
            #     "to": "CR",
            #     "length": edgelen,
            #     # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            # },
            # {
            #     "id": "edge9",
            #     "numLanes": lanes,
            #     "speed": speed_limit,
            #     "from": "CR",
            #     "to": "CL",
            #     "length": edgelen,
            #     # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            # },
            {
                "id": "edge10",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CU",
                "to": "CD",
                "length": 2*edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge11",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CD",
                "to": "CU",
                "length": 2*edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge12",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LU",
                "to": "CL",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge13",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CL",
                "to": "LD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge14",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LD",
                "to": "CD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge15",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CD",
                "to": "RD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge16",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RD",
                "to": "CR",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge17",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CR",
                "to": "RU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge18",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RU",
                "to": "CU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge19",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CU",
                "to": "LU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            }

        ]

        return edges

    def specify_routes(self, net_params):
        rts = {"edge0": ["edge0", "edge1", "edge2", "edge3"],
               "edge1": ["edge1", "edge2", "edge3", "edge0"],
               "edge2": ["edge2", "edge3", "edge0", "edge1"],
               "edge3": ["edge3", "edge0", "edge1", "edge2"],
               "edge4": ["edge4", "edge7", "edge6", "edge5"],
               "edge5": ["edge5", "edge4", "edge7", "edge6"],
               "edge6": ["edge6", "edge5", "edge4", "edge7"],
               "edge7": ["edge7", "edge6", "edge5", "edge4"],

               "edge10": ["edge10", "edge15", "edge16", "edge17", "edge18"],
               "edge15": ["edge15", "edge16", "edge17", "edge17", "edge18"],
               "edge16": ["edge16", "edge17", "edge18", "edge17", "edge18"],
               "edge17": ["edge17", "edge18", "edge10", "edge17", "edge18"],
               "edge18": ["edge18", "edge10", "edge15", "edge17", "edge18"],

               "edge11": ["edge11", "edge19", "edge12", "edge13", "edge14"],
               "edge19": ["edge19", "edge12", "edge13", "edge14", "edge11"],
               "edge12": ["edge12", "edge13", "edge14", "edge11", "edge19"],
               "edge13": ["edge13", "edge14", "edge11", "edge19", "edge12"],
               "edge14": ["edge14", "edge11", "edge19", "edge12", "edge13"],
               }

        return rts


vehicles = VehicleParams()
vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=22)

sim_params = SumoParams(sim_step=0.1, render=True)

initial_config = InitialConfig(spacing="random", bunching=40)

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
net_params = NetParams(additional_params=additional_net_params)

flow_params = dict(
    exp_tag='test_network',
    env_name=AccelEnv,
    network=myNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

# number of time steps
flow_params['env'].horizon = 1500
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1)
