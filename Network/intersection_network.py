from flow.networks import Network


class IntersectionNetwork(Network):
    pass


class IntersectionNetwork(IntersectionNetwork):  # update my network class

    def specify_nodes(self, net_params):
        # one of the elements net_params will need is a "radius" value
        r = net_params.additional_params["length"]

        # specify the name and position (x,y) of each node
        nodes = [{"id": "LU", "x": -r,  "y": +r},  # 1
                 {"id": "RU",  "x": +r,  "y": +r},  # 2
                 {"id": "LD",    "x": -r,  "y": -r},  # 3
                 {"id": "RD",   "x": +r, "y": -r},  # 4
                 {"id": "CL",   "x": -r, "y": 0},  # 5
                 {"id": "CR",   "x": +r, "y": 0},  # 6
                 {"id": "CU",   "x": 0, "y": r},  # 7
                 {"id": "CD",   "x": 0, "y": -r},  # 8
                 {"id": "IT",   "x": 0, "y": 0}]  # 9

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
                "id": "edge0",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LU",
                "to": "CL",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(-pi/2, 0, 40)]
            },
            {
                "id": "edge1",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CL",
                "to": "LD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(0, pi/2, 40)]
            },
            {
                "id": "edge2",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LD",
                "to": "CD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi/2, pi, 40)]
            },
            {
                "id": "edge3",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CD",
                "to": "RD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge4",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RD",
                "to": "CR",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(-pi/2, 0, 40)]
            },
            {
                "id": "edge5",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CR",
                "to": "RU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(0, pi/2, 40)]
            },
            {
                "id": "edge6",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RU",
                "to": "CU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi/2, pi, 40)]
            },
            {
                "id": "edge7",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CU",
                "to": "LU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge8",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LU",
                "to": "CU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge9",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CU",
                "to": "RU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge10",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RU",
                "to": "CR",
                "length": 2*edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge11",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CR",
                "to": "RD",
                "length": 2*edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge12",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "RD",
                "to": "CD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge13",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CD",
                "to": "LD",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge14",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "LD",
                "to": "CL",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
            {
                "id": "edge15",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "CL",
                "to": "LU",
                "length": edgelen,
                # "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            },
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
        rts = {"edge0": [(["edge0", "edge1", "edge2", "edge3", "edge4", "edge5", "edge6", "edge7"], 0.5), (["edge0", "edge1", "edge2", "edge18", "edge22", "edge7"], 0.5)],
               "edge1": [(["edge1", "edge2", "edge3", "edge4", "edge5", "edge6", "edge7", "edge0"], 0.5), (["edge1", "edge2", "edge3", "edge4", "edge17", "edge21"], 0.5)],
               "edge2": [(["edge2", "edge3", "edge4", "edge5", "edge6", "edge7", "edge0", "edge1"], 0.5), (["edge2", "edge18", "edge22", "edge7", "edge0", "edge1"], 0.5)],
               "edge3": [(["edge3", "edge4", "edge5", "edge6", "edge7", "edge0", "edge1", "edge2"], 0.5), (["edge3", "edge4", "edge17", "edge21", "edge1", "edge2"], 0.5)],
               "edge4": [(["edge4", "edge5", "edge6", "edge7", "edge0", "edge1", "edge2", "edge3"], 0.5), (["edge4", "edge5", "edge6", "edge19", "edge23", "edge3"], 0.5)],
               "edge5": [(["edge5", "edge6", "edge7", "edge0", "edge1", "edge2", "edge3", "edge4"], 0.5), (["edge5", "edge6", "edge7", "edge0", "edge16", "edge20"], 0.5)],
               "edge6": [(["edge6", "edge7", "edge0", "edge1", "edge2", "edge3", "edge4", "edge5"], 0.5), (["edge6", "edge19", "edge23", "edge3", "edge4", "edge5"], 0.5)],
               "edge7": [(["edge7", "edge0", "edge1", "edge2", "edge3", "edge4", "edge5", "edge6"], 0.5), (["edge7", "edge0", "edge16", "edge20", "edge5", "edge6"], 0.5)],

               "edge8": [(["edge8", "edge9", "edge10", "edge11", "edge12", "edge13", "edge14", "edge15"], 0.5), (["edge8", "edge19", "edge23", "edge13", "edge14", "edge15"], 0.5)],
               "edge9": [(["edge9", "edge10", "edge11", "edge12", "edge13", "edge14", "edge15", "edge8"], 0.5), (["edge9", "edge10", "edge17", "edge21", "edge15", "edge8"], 0.5)],
               "edge10": [(["edge10", "edge11", "edge12", "edge13", "edge14", "edge15", "edge8", "edge9"], 0.5), (["edge10", "edge11", "edge12", "edge18", "edge22", "edge9"], 0.5)],
               "edge11": [(["edge11", "edge12", "edge13", "edge14", "edge15", "edge8", "edge9", "edge10"], 0.5), (["edge11", "edge12", "edge18", "edge22", "edge9", "edge10"], 0.5)],
               "edge12": [(["edge12", "edge13", "edge14", "edge15", "edge8", "edge9", "edge10", "edge11"], 0.5), (["edge12", "edge13", "edge14", "edge16", "edge20", "edge11"], 0.5)],
               "edge13": [(["edge13", "edge14", "edge15", "edge8", "edge9", "edge10", "edge11", "edge12"], 0.5), (["edge13", "edge14", "edge15", "edge8", "edge19", "edge23"], 0.5)],
               "edge14": [(["edge14", "edge15", "edge8", "edge9", "edge10", "edge11", "edge12", "edge13"], 0.5), (["edge14", "edge16", "edge20", "edge11", "edge12", "edge13"], 0.5)],
               "edge15": [(["edge15", "edge8", "edge9", "edge10", "edge11", "edge12", "edge13", "edge14"], 0.5), (["edge15", "edge8", "edge9", "edge10", "edge17", "edge21"], 0.5)],

               "edge16": [(["edge16", "edge20", "edge11", "edge12", "edge13", "edge14"], 0.5), (["edge16", "edge20", "edge5", "edge6", "edge7", "edge0"], 0.5)],
               "edge17": [(["edge17", "edge21", "edge15", "edge8", "edge9", "edge10"], 0.5), (["edge17", "edge21", "edge1", "edge2", "edge3", "edge4"], 0.5)],

               "edge18": [(["edge18", "edge22", "edge9", "edge10", "edge11", "edge12"], 0.5), (["edge18", "edge22", "edge7", "edge0", "edge1", "edge2"], 0.5)],
               "edge19": [(["edge19", "edge23", "edge13", "edge14", "edge15", "edge8"], 0.5), (["edge19", "edge23", "edge3", "edge4", "edge5", "edge6"], 0.5)],

               "edge20": [(["edge20", "edge5", "edge6", "edge7", "edge0", "edge16"], 0.5), (["edge20", "edge11", "edge12", "edge13", "edge14", "edge16"], 0.5)],
               "edge21": [(["edge21", "edge15", "edge8", "edge9", "edge10", "edge17"], 0.5), (["edge21", "edge1", "edge2", "edge3", "edge4", "edge17"], 0.5)],

               "edge22": [(["edge22", "edge9", "edge10", "edge11", "edge12", "edge18"], 0.5), (["edge22", "edge7", "edge0", "edge1", "edge2", "edge18"], 0.5)],
               "edge23": [(["edge23", "edge13", "edge14", "edge15", "edge8", "edge19"], 0.5), (["edge23", "edge3", "edge4", "edge5", "edge6", "edge19"], 0.5)],

               }

        return rts
