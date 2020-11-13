from flow.networks import Network

class LanechangeNetwork(Network):
    pass

ADDITIONAL_NET_PARAMS={
    "intersection_length": 60,
    "num_lanes":3,
    "speed_limit":30,
}

class LanechangeNetwork(LanechangeNetwork):
    def specify_nodes(self,net_params):
        after_intersection_length=net_params.additional_params['intersection_length']
        nodes=[{'id':'before_intersection','x':-600,'y':0},
        {'id':'center','x':0,'y':0},
        {'id':'after_intersection_up','x':0,'y':after_intersection_length},
        {'id':'after_intersection_down','x':0,'y': -(after_intersection_length)},
        {'id':'after_intersection_right','x':after_intersection_length,'y':0}]
        return nodes

    def specify_edges(self,net_params):
        after_intersection_length=net_params.additional_params['intersection_length']
        lanes=net_params.additional_params['num_lanes']
        edges=[
            {
                'id':'left_intersection',
                'numLanes':lanes,
                'from':'before_intersection',
                'to':'center',
                'length':600,
            },
            {
                'id':'up_intersection',
                'numLanes':lanes,
                'from':'center',
                'to':'after_intersection_up',
                'length':after_intersection_length,
            },
            {
                'id':'up_intersection_back',
                'numLanes':lanes,
                'from':'after_intersection_up',
                'to':'center',
                'length':after_intersection_length,
            },
            {
                'id':'right_intersection',
                'numLanes':lanes,
                'from':'center',
                'to':'after_intersection_right',
                'length':after_intersection_length,
            },
            {
                'id':'down_intersection',
                'numLanes':lanes,
                'from':'center',
                'to':'after_intersection_down',
                'length':after_intersection_length,
            },
            {
                'id':'down_intersection_back',
                'numLanes':lanes,
                'from':'after_intersection_down',
                'to':'center',
                'length':after_intersection_length,
            }
        ]
        return edges

    def specify_connections(self,net_params):
        conn=[]
        lanes=net_params.additional_params['num_lanes']
        for i in range(lanes):
            conn+=[{
                'from':'left_intersection',
                'to':'right_intersection',
                'fromLane':i,
                'toLane':i,
            }]
        conn+=[{
                'from':'left_intersection',
                'to':'up_intersection',
                'fromLane':2,
                'toLane':2,
            }]
        conn+=[{
                'from':'left_intersection',
                'to':'down_intersection',
                'fromLane':0,
                'toLane':0,
            }]    
        return conn
    
    def specify_routes(self,net_params):
        rts={
            'left_intersection':['left_intersection'
                # (['left_intersection','right_intersection'],0.6),
                # (['left_intersection','up_intersection'],0.2),
                # (['left_intersection','down_intersection'],0.2)
            ],
            'right_intersection':['right_intersection'],
            'down_intersection':['down_intersection'],
            'up_intersection':['up_intersection'],
            'down_intersection_back':['down_intersection_back'],
            'up_intersection_back':['up_intersection_back'],
            # 'human_left':[
            #     (['left_intersection','up_intersection'],0.4),
            #     (['left_intersection','right_intersection'],0.6)
            #     ],
            'human_center':[
                (['left_intersection','right_intersection'],1)
                ],
            # 'human_right':[
            #     (['left_intersection','down_intersection'],0.4),
            #     (['left_intersection','right_intersection'],0.6)
            #     ]
        }
        return rts

    # def gen_custom_start_pos(self,cls,net_params,initial_config,num_vehicles):
    #     # inital point of vehicle   
    #     x0 = 6  # position of the first car
    #     dx = 25  # distance between each car
    #     start_lanes = []
    #     start_pos= []
    #     for k in range(num_vehicles):
    #         if k<=20:
    #             start_pos+=[('left_intersection',x0+k*dx)]
    #             start_lanes+=[2]
    #         elif k<=40:
    #             start_pos+=[('left_intersection',x0+(k-20)*dx)]
    #             start_lanes+=[1]
    #         else:
    #             start_pos+=[('left_intersection',x0+(k-40)*dx)]
    #             start_lanes+=[0]


    #     return start_pos, start_lanes