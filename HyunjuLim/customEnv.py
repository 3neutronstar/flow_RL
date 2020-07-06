# tutorial_08: Creating Custom Environments

# we will create an environment in which the accelerations of a handful of vehicles in the network are specified by a single centralized agent, with the objective of the agent being to improve the average speed of all vehicle in the network.

# import the base environment class
from flow.envs import Env

# define the environment class, and inherit properties from the base environment class
class myEnv(Env):
    pass
