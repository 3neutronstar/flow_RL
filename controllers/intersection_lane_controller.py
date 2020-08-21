from flow.controllers.base_lane_changing_controller import BaseLaneChangeController


class IntersectionLaneChangeController(BaseLaneChangeController):
    def __init__(self, veh_id, lane_change_params=None):
        if lane_change_params is None:
            lane_change_params = {}

        self.veh_id = veh_id
        self.lane_change_params = lane_change_params

    def get_lane_change_action(self, env):

        return lane_change_action

    def get_action(self, env):
        lc_action = self.get_lane_change_action(env)
        return lc_action
