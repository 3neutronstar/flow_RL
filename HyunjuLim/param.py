class SimParams(object):
    """Simulation-specific parameters.

    All subsequent parameters of the same type must extend this.

    Attributes
    ----------
    sim_step : float optional
        seconds per simulation step; 0.1 by default
    render : str or bool, optional
        specifies whether to visualize the rollout(s)

        * False: no rendering
        * True: delegate rendering to sumo-gui for back-compatibility
        * "gray": static grayscale rendering, which is good for training
        * "dgray": dynamic grayscale rendering
        * "rgb": static RGB rendering
        * "drgb": dynamic RGB rendering, which is good for visualization

    restart_instance : bool, optional
        specifies whether to restart a simulation upon reset. Restarting
        the instance helps avoid slowdowns cause by excessive inflows over
        large experiment runtimes, but also require the gui to be started
        after every reset if "render" is set to True.
    emission_path : str, optional
        Path to the folder in which to create the emissions output.
        Emissions output is not generated if this value is not specified
    save_render : bool, optional
        specifies whether to save rendering data to disk
    sight_radius : int, optional
        sets the radius of observation for RL vehicles (meter)
    show_radius : bool, optional
        specifies whether to render the radius of RL observation
    pxpm : int, optional
        specifies rendering resolution (pixel / meter)
    force_color_update : bool, optional
        whether or not to automatically color vehicles according to their types
    """

class AimsunParams(SimParams):
    """Aimsun-specific simulation parameters.

    Extends SimParams.

    Attributes
    ----------
    sim_step : float optional
        seconds per simulation step; 0.1 by default
    render : str or bool, optional
        specifies whether to visualize the rollout(s)

        * False: no rendering
        * True: delegate rendering to sumo-gui for back-compatibility
        * "gray": static grayscale rendering, which is good for training
        * "dgray": dynamic grayscale rendering
        * "rgb": static RGB rendering
        * "drgb": dynamic RGB rendering, which is good for visualization

    restart_instance : bool, optional
        specifies whether to restart a simulation upon reset. Restarting
        the instance helps avoid slowdowns cause by excessive inflows over
        large experiment runtimes, but also require the gui to be started
        after every reset if "render" is set to True.
    emission_path : str, optional
        Path to the folder in which to create the emissions output.
        Emissions output is not generated if this value is not specified
    save_render : bool, optional
        specifies whether to save rendering data to disk
    sight_radius : int, optional
        sets the radius of observation for RL vehicles (meter)
    show_radius : bool, optional
        specifies whether to render the radius of RL observation
    pxpm : int, optional
        specifies rendering resolution (pixel / meter)
    network_name : str, optional
        name of the network generated in Aimsun.
    experiment_name : str, optional
        name of the experiment generated in Aimsun
    replication_name : str, optional
        name of the replication generated in Aimsun. When loading
        an Aimsun template, this parameter must be set to the name
        of the replication to be run by the simulation; in this case,
        the network_name and experiment_name parameters are not
        necessary as they will be obtained from the replication name.
    centroid_config_name : str, optional
        name of the centroid configuration to load in Aimsun. This
        parameter is only used when loading an Aimsun template,
        not when generating one.
    subnetwork_name : str, optional
        name of the subnetwork to load in Aimsun. This parameter is not
        used when generating a network; it can be used when loading an
        Aimsun template containing a subnetwork in order to only load
        the objects contained in this subnetwork. If set to None or if the
        specified subnetwork does not exist, the whole network will be loaded.
    """
    class SumoParams(SimParams):
    """Sumo-specific simulation parameters.

    Extends SimParams.

    These parameters are used to customize a sumo simulation instance upon
    initialization. This includes passing the simulation step length,
    specifying whether to use sumo's gui during a run, and other features
    described in the Attributes below.

    Attributes
    ----------
    port : int, optional
        Port for Traci to connect to; finds an empty port by default
    sim_step : float optional
        seconds per simulation step; 0.1 by default
    emission_path : str, optional
        Path to the folder in which to create the emissions output.
        Emissions output is not generated if this value is not specified
    lateral_resolution : float, optional
        width of the divided sublanes within a lane, defaults to None (i.e.
        no sublanes). If this value is specified, the vehicle in the
        network cannot use the "LC2013" lane change model.
    no_step_log : bool, optional
        specifies whether to add sumo's step logs to the log file, and
        print them into the terminal during runtime, defaults to True
    render : str or bool, optional
        specifies whether to visualize the rollout(s)

        * False: no rendering
        * True: delegate rendering to sumo-gui for back-compatibility
        * "gray": static grayscale rendering, which is good for training
        * "dgray": dynamic grayscale rendering
        * "rgb": static RGB rendering
        * "drgb": dynamic RGB rendering, which is good for visualization

    save_render : bool, optional
        specifies whether to save rendering data to disk
    sight_radius : int, optional
        sets the radius of observation for RL vehicles (meter)
    show_radius : bool, optional
        specifies whether to render the radius of RL observation
    pxpm : int, optional
        specifies rendering resolution (pixel / meter)
    force_color_update : bool, optional
        whether or not to automatically color vehicles according to their types
    overtake_right : bool, optional
        whether vehicles are allowed to overtake on the right as well as
        the left
    seed : int, optional
        seed for sumo instance
    restart_instance : bool, optional
        specifies whether to restart a sumo instance upon reset. Restarting
        the instance helps avoid slowdowns cause by excessive inflows over
        large experiment runtimes, but also require the gui to be started
        after every reset if "render" is set to True.
    print_warnings : bool, optional
        If set to false, this will silence sumo warnings on the stdout
    teleport_time : int, optional
        If negative, vehicles don't teleport in gridlock. If positive,
        they teleport after teleport_time seconds
    num_clients : int, optional
        Number of clients that will connect to Traci
    color_by_speed : bool
        whether to color the vehicles by the speed they are moving at the
        current time step
    use_ballistic: bool, optional
        If true, use a ballistic integration step instead of an euler step
    """

    class EnvParams:
    """Environment and experiment-specific parameters.

    This includes specifying the bounds of the action space and relevant
    coefficients to the reward function, as well as specifying how the
    positions of vehicles are modified in between rollouts.

    Attributes
    ----------
    additional_params : dict, optional
        Specify additional environment params for a specific
        environment configuration
    horizon : int, optional
        number of steps per rollouts
    warmup_steps : int, optional
        number of steps performed before the initialization of training
        during a rollout. These warmup steps are not added as steps
        into training, and the actions of rl agents during these steps
        are dictated by sumo. Defaults to zero
    sims_per_step : int, optional
        number of sumo simulation steps performed in any given rollout
        step. RL agents perform the same action for the duration of
        these simulation steps.
    evaluate : bool, optional
        flag indicating that the evaluation reward should be used
        so the evaluation reward should be used rather than the
        normal reward
    clip_actions : bool, optional
        specifies whether to clip actions from the policy by their range when
        they are inputted to the reward function. Note that the actions are
        still clipped before they are provided to `apply_rl_actions`.
    """

    class NetParams:
    """Network configuration parameters.

    Unlike most other parameters, NetParams may vary drastically dependent
    on the specific network configuration. For example, for the ring road
    the network parameters will include a characteristic length, number of
    lanes, and speed limit.

    In order to determine which additional_params variable may be needed
    for a specific network, refer to the ADDITIONAL_NET_PARAMS variable
    located in the network file.

    Attributes
    ----------
    inflows : InFlows type, optional
        specifies the inflows of specific edges and the types of vehicles
        entering the network from these edges
    osm_path : str, optional
        path to the .osm file that should be used to generate the network
        configuration files
    template : str, optional
        path to the network template file that can be used to instantiate a
        netowrk in the simulator of choice
    additional_params : dict, optional
        network specific parameters; see each subclass for a description of
        what is needed
    """

    class InitialConfig:
    """Initial configuration parameters.

    These parameters that affect the positioning of vehicle in the
    network at the start of a rollout. By default, vehicles are uniformly
    distributed in the network.

    Attributes
    ----------
    shuffle : bool, optional  # TODO: remove
        specifies whether the ordering of vehicles in the Vehicles class
        should be shuffled upon initialization.
    spacing : str, optional
        specifies the positioning of vehicles in the network relative to
        one another. May be one of: "uniform", "random", or "custom".
        Default is "uniform".
    min_gap : float, optional  # TODO: remove
        minimum gap between two vehicles upon initialization, in meters.
        Default is 0 m.
    x0 : float, optional  # TODO: remove
        position of the first vehicle to be placed in the network
    perturbation : float, optional
        standard deviation used to perturb vehicles from their uniform
        position, in meters. Default is 0 m.
    bunching : float, optional
        reduces the portion of the network that should be filled with
        vehicles by this amount.
    lanes_distribution : int, optional
        number of lanes vehicles should be dispersed into. If the value is
        greater than the total number of lanes on an edge, vehicles are
        spread across all lanes.
    edges_distribution : str or list of str or dict, optional
        edges vehicles may be placed on during initialization, may be one
        of:

        * "all": vehicles are distributed over all edges
        * list of edges: list of edges vehicles can be distributed over
        * dict of edges: where the key is the name of the edge to be
          utilized, and the elements are the number of cars to place on
          each edge
    additional_params : dict, optional
        some other network-specific params
    """
