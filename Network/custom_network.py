from flow.networks import Network
from flow.core.params import NetParams
import numpy as np
import sys
import queue


class Custom_Network(Network):
    pass


ADDITIONAL_NET_PARAMS = {
    "length": 150,
    "lanes": 1,
    "speed_limit": 30,
    "resolution": 40,
}
net_params = NetParams(
    additional_params=ADDITIONAL_NET_PARAMS)


class Custom_Network(Custom_Network):
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
