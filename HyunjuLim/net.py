from flow.networks import Network
import numpy as np
import sys
import queue


class Custom_Network(Network):
    pass

ADDITIONAL_NET_PARAMS = {
    "st_line": 40,
    "num_lanes": 1,
    "speed_limit": 30,
}

class Custom_Network(Custom_Network):
    def specify_nodes(self, net_params):
        
        r = net_params.additional_params["st_line"]

        return nodes

    def specify_edges(self, net_params):
        #
        return edges

    def specify_edge_starts(self, net_params):
        #

        return

    def specify_routes(self, net_params):
        #
        return routes