from flow.controllers import IDMController
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, SumoLaneChangeParams
from flow.core.params import VehicleParams, InFlows
from flow.envs.ring.lane_change_accel import ADDITIONAL_ENV_PARAMS
from Network.lanechange_network import LanechangeNetwork, ADDITIONAL_NET_PARAMS
from flow.envs import LaneChangeAccelEnv
from flow.controllers.base_routing_controller import BaseRouter
from flow.controllers import ContinuousRouter
from flow.controllers.lane_change_controllers import SimLaneChangeController
from flow.controllers.base_lane_changing_controller import BaseLaneChangeController
import random
class LanechangeRouter(BaseRouter):
    def choose_route(self,env):
        veh_id=self.veh_id
        vehicles = env.k.vehicle
        veh_type=vehicles.get_type(veh_id)
        veh_edge = vehicles.get_edge(veh_id) #route
        veh_route=vehicles.get_route(veh_id)
        if len(env.k.vehicle.get_route(self.veh_id)) == 0:
            return None

        elif veh_edge=='left_intersection'and veh_route[-1]=='left_intersection':
            random_num=random.random()
            if random_num<=0.8:
                if veh_type[6]=='r':
                    next_route=(veh_edge,'down_intersection')
                elif veh_type[6]=='c':
                    next_route=(veh_edge,'right_intersection')
                elif veh_type[6]=='l':
                    next_route=(veh_edge,'up_intersection')
            else:
                next_route=(veh_edge,'right_intersection')
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
        else:
            return None
            # next_route=[vehicles.get_edge(veh_id)]
        print(veh_id,", ",veh_type,": ",veh_edge)
        print(next_route,",,,",veh_route)
        return next_route

# class LanechangeController(BaseLaneChangeController):
#     def

vehicles = VehicleParams()
vehicles.add(
    veh_id="human_left",
    acceleration_controller=(IDMController, {}),
    lane_change_params=SumoLaneChangeParams(
        model="SL2015",
        lc_sublane=2.0,
    ),
    routing_controller=(LanechangeRouter,{}),
    lane_change_controller=(SimLaneChangeController,{}),
    num_vehicles=0)
vehicles.add(
    veh_id="human_center",
    acceleration_controller=(IDMController, {}),
    lane_change_params=SumoLaneChangeParams(
        model="SL2015",
        lc_sublane=2.0,
    ),
    lane_change_controller=(SimLaneChangeController,{}),
    routing_controller=(LanechangeRouter,{}),
    num_vehicles=0)    
vehicles.add(
    veh_id="human_right",
    acceleration_controller=(IDMController, {}),
    lane_change_params=SumoLaneChangeParams(
        model="SL2015",
        lc_sublane=2.0,
    ),
    routing_controller=(LanechangeRouter,{}),
    lane_change_controller=(SimLaneChangeController,{}),
    num_vehicles=0)

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

inflow = InFlows()
inflow.add(
    veh_type="human_left",
    edge="left_intersection",
    probability=0.25,
    departLane=2,
    departSpeed=20)
inflow.add(
    veh_type="human_center",
    edge="left_intersection",
    probability=0.25,
    departLane=1,
    departSpeed=20)
inflow.add(
    veh_type="human_right",
    edge="left_intersection",
    probability=0.25,
    departLane=0,
    departSpeed=20)

flow_params = dict(
    # name of the experiment
    exp_tag='Lanechange',

    # name of the flow environment the experiment is running on
    env_name=LaneChangeAccelEnv,

    # name of the network class the experiment is running on
    network=LanechangeNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        lateral_resolution=1.0,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=2000,
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        shuffle=True,
    ),
)