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
    "num_lanes": 2,
    "speed_limit": 30,
}


class myNetwork(myNetwork):  # update my network class
    def specify_nodes(self, net_params):
        # one of the elements net_params will need is a "radius" value
        r = net_params.additional_params["st_line"]

        # specify the name and position (x,y) of each node
        nodes = [{"id": "LeftUp", "x": -r,  "y": +r},
                 {"id": "CenterUp",  "x": +0,  "y": +r},
                 {"id": "LeftDown",    "x": -r,  "y": 0},
                 {"id": "CenterDown",   "x": +0, "y": 0},
                 {"id": "RightDown",    "x": +r,  "y": 0},
                 {"id": "RightUp",   "x": +r, "y": +r}
                 ]

        return nodes

    def specify_edges(self, net_params):
        r = net_params.additional_params["st_line"]
        edgelen = r
        # this will let us control the number of lanes in the network
        lanes = net_params.additional_params["num_lanes"]
        # speed limit of vehicles in the network
        speed_limit = net_params.additional_params["speed_limit"]

        edges = [
            {
                "id": "edge0",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LeftUp",
                "to": "LeftDown",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(-pi/2, 0, 40)]
            },
            {
                "id": "edge1",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LeftDown",
                "to": "CenterDown",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(0, pi/2, 40)]
            },
            {
                "id": "edge2",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CenterDown",
                "to": "CenterUp",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi/2, pi, 40)]
            },
            {
                "id": "edge3",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CenterUp",
                "to": "LeftUp",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge4",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LeftDown",
                "to": "LeftUp",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(-pi/2, 0, 40)]
            },
            {
                "id": "edge5",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CenterDown",
                "to": "LeftDown",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(0, pi/2, 40)]
            },
            {
                "id": "edge6",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CenterUp",
                "to": "CenterDown",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi/2, pi, 40)]
            },
            {
                "id": "edge7",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LeftUp",
                "to": "CenterUp",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge8",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CenterUp",
                "to": "RightUp",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(0, pi/2, 40)]
            },
            {
                "id": "edge9",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RightUp",
                "to": "RightDown",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi/2, pi, 40)]
            },
            {
                "id": "edge10",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RightDown",
                "to": "CenterDown",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge11",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CenterDown",
                "to": "RightDown",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(0, pi/2, 40)]
            },
            {
                "id": "edge12",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RightDown",
                "to": "RightUp",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi/2, pi, 40)]
            },
            {
                "id": "edge13",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RightUp",
                "to": "CenterUp",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            }
        ]

        return edges

    def specify_routes(self, net_params):
        rts = {"edge0": [(["edge0", "edge1", "edge2", "edge3"], 0.5), (["edge0", "edge1", "edge11", "edge12", "edge13", "edge3"], 0.5)],
               "edge1": [(["edge1", "edge2", "edge3", "edge0"], 0.5), (["edge1", "edge11", "edge12", "edge13", "edge3", "edge0"], 0.5)],
               "edge2": [(["edge2", "edge3", "edge0", "edge1"], 0.5), (["edge2", "edge8", "edge9", "edge10"], 0.5)],
               "edge3": [(["edge3", "edge0", "edge1", "edge2"], 0.5), (["edge3", "edge0", "edge1", "edge11", "edge12", "edge13"], 0.5)],
               "edge4": [(["edge4", "edge7", "edge6", "edge5"], 0.5), (["edge4", "edge7", "edge8", "edge9", "edge10", "edge5"], 0.5)],
               "edge5": [(["edge5", "edge4", "edge7", "edge6"], 0.5), (["edge5", "edge4", "edge7", "edge8", "edge9", "edge10"], 0.5)],
               "edge6": [(["edge6", "edge5", "edge4", "edge7"], 0.5), (["edge6", "edge11", "edge12", "edge13"], 0.5)],
               "edge7": [(["edge7", "edge6", "edge5", "edge4"], 0.5), (["edge7", "edge8", "edge9", "edge10", "edge5", "edge4"], 0.5)],
               "edge8": [(["edge8", "edge9", "edge10", "edge2"], 0.5), (["edge8", "edge9", "edge10","edge5", "edge4", "edge7"], 0.5)],
               "edge9": [(["edge9", "edge10", "edge2", "edge8"], 0.5), (["edge9", "edge10","edge5", "edge4", "edge7", "edge8"], 0.5)],
               "edge10": [(["edge10", "edge2", "edge8", "edge9"], 0.5), (["edge10","edge5", "edge4", "edge7", "edge8", "edge9"], 0.5)],
               "edge11": [(["edge11", "edge12", "edge13", "edge6"], 0.5), (["edge11", "edge12", "edge13","edge3", "edge0", "edge1"], 0.5)],
               "edge12": [(["edge12", "edge13", "edge6", "edge11"], 0.5), (["edge12", "edge13","edge3", "edge0", "edge1", "edge11"], 0.5)],
               "edge13": [(["edge13", "edge6", "edge11", "edge12"], 0.5), (["edge13","edge3", "edge0", "edge1", "edge11", "edge12"], 0.5)],
               }
        # rts = { "edge0": [(["edge1"],1)],
        #         "edge1": [(["edge2"],0.5), (["edge11"],0.5)],
        #         "edge2": [(["edge3"],0.5), (["edge8"],0.5)],
        #         "edge3": [(["edge0"],1)],
        #         "edge4": [(["edge7"],1)],
        #         "edge5": [(["edge4"],1)],
        #         "edge6": [(["edge11"],0.5), (["edge5"],0.5)],
        #         "edge7": [(["edge6"],0.5), (["edge8"],0.5)],
        #         "edge8": [(["edge9"],1)],
        #         "edge9": [(["edge10"],1)],
        #         "edge10": [(["edge2"],0.5), (["edge5"],0.5)],
        #         "edge11": [(["edge12"],1)],
        #         "edge12": [(["edge13"],1)],
        #         "edge13": [(["edge3"],0.5), (["edge6"],0.5)]
        #     }

        return rts


vehicles = VehicleParams()
vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=22)


sim_params = SumoParams(sim_step=0.1, render=True)

initial_config = InitialConfig(bunching=40)

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
