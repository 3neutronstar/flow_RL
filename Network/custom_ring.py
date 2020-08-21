from flow.networks import Network
import numpy as np
import sys
import queue


ADDITIONAL_NET_PARAMS = {
    "length": 50,
    "lanes": 2,
    "speed_limit": 30,
    "resolution": 40
}


class RingNetwork_custom(Network):
    pass


class RingNetwork_custom(RingNetwork_custom):

    def specify_nodes(self, net_params):
        r = ADDITIONAL_NET_PARAMS["length"]
        nodes = [{"id": "0", "x": 0, "y": +r},
                 {"id": "1", "x": +r, "y": +r},
                 {"id": "2", "x": +r, "y": 0},
                 {"id": "3", "x": 0, "y": 0}
                 ]
        return nodes

    def specify_edges(self, net_params):
        damg = np.array([[0, 2, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 2],
                         [2, 0, 0, 0]])
        length = ADDITIONAL_NET_PARAMS["length"]
        resolution = ADDITIONAL_NET_PARAMS["resolution"]
        edgelen = length
        # this will let us control the number of lanes in the network
        #lanes = net_params.additional_params["num_lanes"]
        # speed limit of vehicles in the network
        speed_limit = ADDITIONAL_NET_PARAMS["speed_limit"]
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
        rts = {"e_0_1": ["e_0_1", "e_1_2"],
               "e_1_2": ["e_1_2", "e_2_3"],
               "e_2_3": ["e_2_3", "e_3_0"],
               "e_3_0": ["e_3_0", "e_0_1"]}
        #  rts = {"e_0_1": ["e_0_1", "e_1_2", "e_2_3", "e_3_0"],
        #        "e_1_2": ["e_1_2", "e_2_3", "e_3_0", "e_0_1"],
        #        "e_2_3": ["e_2_3", "e_3_0", "e_0_1", "e_1_2"],
        #        "e_3_0": ["e_3_0", "e_0_1", "e_1_2", "e_2_3"]}

        return rts

    # def specify_routes(self, net_params):
    #     damg = np.array([[0, 1, 0, 0],
    #                      [0, 0, 1, 0],
    #                      [0, 0, 0, 1],
    #                      [1, 0, 0, 0]])
    #     damg_rows = damg.shape[0]
    #     damg_cols = damg.shape[1]

    #     route = np.zeros((damg_rows, damg_cols), dtype=object)

    #     for n_rows in range(0, damg_cols):
    #         for n_cols in range(0, damg_cols):
    #             if(damg[n_rows][n_cols] > 0):
    #                 q = queue.Queue(damg_cols)
    #                 route[n_rows][n_cols] = q
    #                 route[n_rows][n_cols].put(
    #                     str("e_"+str(n_rows)+"_"+str(n_cols)))

    #     rts = {}
    #     sum_row = 0
    #     for n_rows in range(0, damg_rows):
    #         for n_cols in range(0, damg_cols):
    #             sum_row += damg[n_rows][n_cols]
    #             if n_rows != n_cols:
    #                 route_array = []
    #                 if route[n_rows][n_cols] != 0:
    #                     # while route[n_rows][n_cols].empty() or route[n_rows][n_cols] != 0:
    #                     while route[n_rows][n_cols].qsize():
    #                         route_array.append(
    #                             route[n_rows][n_cols].get_nowait())
    #                 start_edge = str("e_"+str(n_rows)+"_"+str(n_cols))

    #                 rts[start_edge] = [(route_array)]
    #             else:
    #                 continue
    #     return rts
