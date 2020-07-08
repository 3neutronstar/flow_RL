from flow.networks import Network
import numpy as np
from numpy import pi, sin, cos, linspace


class RingNetwork_intest(Network):
    pass


ADDITIONAL_NET_PARAMS = {
    "length": 40,
    "num_lanes": 1,
    "speed_limit": 30,
}


class RingNetwork_intest(RingNetwork_intest):
    def specify_nodes(self, net_params):
        r = net_params.additional_params["length"]
        nodes = [{"id": "0", "x": 0, "y": +r},
                 {"id": "1", "x": +r, "y": +r},
                 {"id": "2", "x": +r, "y": 0},
                 {"id": "3", "x": 0, "y": 0}
                 ]
        return nodes

    def specify_edges(self, net_params):
        damg = np.array([[0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [1, 0, 0, 0]])
        length = net_params.additional_params["length"]
        resolution = net_params.additional_params["resolution"]
        edgelen = length
        # this will let us control the number of lanes in the network
        # speed limit of vehicles in the network
        speed_limit = net_params.additional_params["speed_limit"]
        # L: left, R: right, U: Up D:Down
        edges = list()
        # if you want to connect change num lanes, chage the number of damg component
        damg_cols = damg.shape[0]
        damg_rows = damg.shape[1]
        for n_cols in range(0, damg_cols):
            for n_rows in range(0, damg_rows):
                if(damg[n_cols][n_rows] > 0):
                    insert_id = str("e_"+str(n_cols)+"_"+str(n_rows))
                    edges.append({"id": insert_id,
                                  "from": str(n_cols),
                                  "to": str(n_rows),
                                  "numLanes": damg[n_cols][n_rows],
                                  "speed": speed_limit,
                                  "length": edgelen,
                                  # if you want to customize call seperately
                                  })
        return edges

    def specify_routes(self, net_params):
        rts = {"e_0_1": ["e_0_1", "e_1_2", "e_2_3", "e_3_0"],
               "e_1_2": ["e_1_2", "e_2_3", "e_3_0", "e_0_1"],
               "e_2_3": ["e_2_3", "e_3_0", "e_0_1", "e_1_2"],
               "e_3_0": ["e_3_0", "e_0_1", "e_1_2", "e_2_3"]}

        return rts
    # def specify_routes(self, net_params):
    #     damg = np.array([[0, 1, 0, 0],
    #                      [0, 0, 1, 0],
    #                      [0, 0, 0, 1],
    #                      [1, 0, 0, 0]])
    #     damg_cols = damg.shape[0]
    #     damg_rows = damg.shape[1]
    #     import queue
    #     route[damg_cols][damg_rows] = [queue.Queue(damg_cols)]
    #     for n_cols in range(0, damg_cols):
    #         for n_rows in range(0, damg_cols):
    #             if(damg[n_cols][n_rows] > 0):
    #                 route[n_cols][n_rows].put(str("e_"+n_cols+"_"+n_rows))
    #     for n_cols in range(0, damg_cols):
    #         for n_rows in range(0, damg_cols):
    #             sum_row = 0
    #             for idx in range(0, damg_cols):
    #                 sum_row += idx
    #             route_array = []
    #             while np.route[n_cols][n_rows].empty:
    #                 route_array.append(np.route[n_cols][n_rows].get)
    #             start_edge = str("e_"+n_cols+"_"+n_rows)
    #             rts = {start_edge: [(route_array, 1/sum_row)]}
    #     return rts
