from flow.networks import Network
import numpy as np


class Turnleft_Network(Network):
    pass


ADDITIONAL_NET_PARAMS = {
    "length": 50,
    "lanes": 2,
    "speed_limit": 30,
    "resolution": 40,
}


class Turnleft_Network(Turnleft_Network):

    def specify_nodes(self, net_params):

        r = net_params.additional_params["length"]

        nodes = [{"id": "0", "x": -r, "y": 0},
                 {"id": "1", "x": 0, "y": +r},
                 {"id": "2", "x": 0, "y": 0},
                 {"id": "3", "x": 0, "y": -r}]

        return nodes

    def specify_edges(self, net_params):
        damg = np.array([[0, 0, 0, 0],  #
                         [0, 0, 0, 0],  #
                         [2, 2, 0, 0],  # 2->0,2->1
                         [0, 0, 2, 0]])  # 3->2
        length = net_params.additional_params["length"]
        resolution = net_params.additional_params["resolution"]
        edgelen = length
        # this will let us control the number of lanes in the network
        #lanes = net_params.additional_params["num_lanes"]
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
        rts = {
            "e_3_2": [(["e_3_2", "e_2_0"], 0.4), (["e_3_2", "e_2_1"], 0.6)],
            "e_2_0": [(["e_2_0"], 1)],
            "e_2_1": [(["e_2_1"], 1)],
        }
        return rts
