# new file

"""Grid example."""
from flow.controllers import GridRouter

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import InFlows

from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS

from flow.networks import TrafficLightGridNetwork
from flow.networks import Network

from flow.core.params import NetParams


class Traffic_Network(Network):
    pass


ADDITIONAL_NET_PARAMS = {
    "length": 150,
    "lanes": 1,
    "speed_limit": 30,
    "resolution": 40,
}

initial = InitialConfig(
    spacing='custom', lanes_distribution=float('inf'), shuffle=True)
inflow = InFlows()
inflow.add(
    veh_type='human_up',
    edge="edge19",
    probability=0.25,
    departLane='free',
    departSpeed=20)
inflow.add(
    veh_type='human_down',
    edge="edge18",
    probability=0.25,
    departLane='free',
    departSpeed=20)
inflow.add(
    veh_type='human_left',
    edge="edge16",
    probability=0.25,
    departLane='free',
    departSpeed=20)
inflow.add(
    veh_type='human_right',
    edge="edge17",
    probability=0.25,
    departLane='free',
    departSpeed=20)

net_params = NetParams(
    inflows=inflow,
    additional_params=ADDITIONAL_NET_PARAMS)


vehicles = VehicleParams()

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

tl_logic = TrafficLightParams(baseline=False)

phases = [{
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "GrGrGrGrGrGr"
}, {
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "yryryryryryr"
}, {
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "rGrGrGrGrGrG"
}, {
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "ryryryryryry"
}]
tl_logic.add("IT", phases=phases, programID=1)
#tl_logic.add("center1", phases=phases, programID=1)
#tl_logic.add("center2", phases=phases, programID=1, tls_type="actuated")

flow_params = dict(
    # name of the experiment
    exp_tag='grid-intersection',
    # name of the flow environment the experiment is running on
    env_name=AccelEnv,
    # name of the network class the experiment is running on
    network=Traffic_Network,
    # simulator that is used by the experiment
    simulator='traci',
    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=True,
    ),
    # environment related parameters (see flow.core.params.EnvParams)
    env=env_params,
    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,
    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,
    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=initial,
    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    tls=tl_logic,
)


class Traffic_Network(Traffic_Network):  # update my network class

    def specify_nodes(self, net_params):
        # one of the elements net_params will need is a "radius" value
        r = net_params.additional_params["length"]

        # specify the name and position (x,y) of each node
        nodes = [{"id": "IT",   "x": 0, "y": 0, "type": "traffic_light"},  # 9
                 {"id": "CL",   "x": -r, "y": 0, "type": "priority"},  # 5
                 {"id": "CR",   "x": +r, "y": 0, "type": "priority"},  # 6
                 {"id": "CU",   "x": 0, "y": +r, "type": "priority"},  # 7
                 {"id": "CD",   "x": 0, "y": -r, "type": "priority"}]  # 8
        return nodes

    def specify_edges(self, net_params):
        r = net_params.additional_params["length"]
        edgelen = r
        # this will let us control the number of lanes in the network
        lanes = net_params.additional_params["lanes"]
        # speed limit of vehicles in the network
        speed_limit = net_params.additional_params["speed_limit"]
        # L: left, R: right, U: Up D:Down
        edges = [
            {
                "id": "edge16",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CL",
                "to": "IT",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge17",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CR",
                "to": "IT",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge18",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CD",
                "to": "IT",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge19",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CU",
                "to": "IT",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge20",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "IT",
                "to": "CR",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge21",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "IT",
                "to": "CL",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge22",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "IT",
                "to": "CU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge23",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "IT",
                "to": "CD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            }
        ]

        return edges

    def specify_routes(self, net_params):
        rts = {"edge16": [(["edge16", "edge20"], 0.5), (["edge16", "edge22"], 0.5)],
               "edge17": [(["edge17", "edge21"], 0.5), (["edge16", "edge23"], 0.5)],
               "edge18": [(["edge18", "edge22"], 0.5), (["edge18", "edge21"], 0.5)],
               "edge19": [(["edge19", "edge23"], 0.5), (["edge19", "edge20"], 0.5)],
               "edge20": [(["edge20"], 1)],
               "edge21": [(["edge21"], 1)],
               "edge22": [(["edge22"], 1)],
               "edge23": [(["edge23"], 1)]
               }

        return rts
