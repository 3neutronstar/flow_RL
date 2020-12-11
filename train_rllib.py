import argparse
import json
import os
import sys
from time import strftime
from copy import deepcopy
import numpy as np
import timeit
import torch
from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env
from Experiment.experiment import Experiment
import getpass






def parse_args(args):
    """Parse training options user can specify in command line.
    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.RawDescriptionHelpFormatter,
    #     description="Parse argument used when running a Flow simulation.",
    #     epilog="python simulate.py EXP_CONFIG")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python simulate.py EXP_CONFIG")
    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
    )  # Name of the experiment configuration file, as located in
    # exp_configs/non_rl exp_configs/rl/singleagent or exp_configs/rl/multiagent.'

    parser.add_argument(  # for rllib
        '--algorithm', type=str, default="PPO",
    )  # choose algorithm in order to use
    parser.add_argument(
        '--num_cpus', type=int, default=1,
    )  # How many CPUs to usedh r
    parser.add_argument(  # batch size
        '--rollout_size', type=int, default=100,
    )  # How many steps are in a training batch.
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
    )  # Directory with checkpoint to restore training from.
    parser.add_argument(
        '--no_render',
        action='store_true',
    )  # Specifies whether to run the simulation during runtime.
    return parser.parse_known_args(args)[0]


def setup_exps_rllib(flow_params,
                     n_cpus,
                     n_rollouts,
                     policy_graphs=None,
                     policy_mapping_fn=None,
                     policies_to_train=None,
                     flags=None):
    from ray import tune
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class
    import torch
    horizon = flow_params['env'].horizon
    if flags.algorithm.lower() == "ppo":
        alg_run = "PPO"
        agent_cls = get_agent_class(alg_run)
        config = deepcopy(agent_cls._default_config)
        config["num_workers"] = 2
        config["horizon"] = horizon
        config['num_gpus']=1
        config['num_cpus_per_worker']=1
        if flags.exp_config== 'singleagent_ring':
            config["gamma"] = 0.99  # discount rate
            config["use_gae"] = True  # truncated
            config["lambda"] = 0.97  # truncated value
            config["kl_target"] = 0.02  # d_target
            config["num_sgd_iter"] = 15
            config["sgd_minibatch_size"] = 1024
            config['lr']=5e-7 
            config["clip_param"] = 0.2

        elif flags.exp_config=='singleagent_figure_eight':
            config['sgd_minibatch_size']=64
            config["clip_param"] = 0.2
            #Exploration
            config['exploration_config']["type"] = "GaussianNoise"
            config['exploration_config']["initial_scale"] = 1.0
            config['exploration_config']["final_scale"] = 0.02
            config['exploration_config']["scale_timesteps"] = 1000000
            config['exploration_config']["random_timesteps"] = 1000
            config['exploration_config']["stddev"] = 0.1
        

    elif flags.algorithm.lower() == "ddpg":
        from ray.rllib.agents.ddpg.ddpg import DEFAULT_CONFIG
        alg_run = "DDPG"
        agent_cls = get_agent_class(alg_run)
        config = deepcopy(agent_cls._default_config)
        config["num_workers"] = 1
        # config['num_gpus']=1
        # model
        if flags.exp_config == 'singleagent_ring':
            config['n_step'] = 1
            config['actor_hiddens'] = [64, 64]
            config['actor_lr'] = 0.0001  # in article 'ddpg'
            config['critic_lr'] = 0.0001
            config['critic_hiddens'] = [64, 64]
            config['gamma'] = 0.99
            config['model']['fcnet_hiddens'] = [128, 128]
            config['lr']=1e-4
            # exploration
            config['exploration_config']['final_scale'] = 0.02
            config['exploration_config']['scale_timesteps'] = 2100000
            config['exploration_config']['ou_base_scale'] = 0.1
            config['exploration_config']['ou_theta'] = 0.15
            config['exploration_config']['ou_sigma'] = 0.2
            # optimization
            config['tau'] = 0.001
            config['l2_reg'] = 1e-6
            config['train_batch_size'] = 64
            config['learning_starts'] = 3000
            # evaluation
            # config['evaluation_interval'] = 5
            config['buffer_size'] = 300000 #3e5
            config['timesteps_per_iteration'] = 3000
            config['prioritized_replay']=False
            config["prioritized_replay_beta_annealing_timesteps"]=2200000
            config['final_prioritized_replay_beta']=0.01


        elif flags.exp_config == 'singleagent_figure_eight':
            config['n_step'] = 3
            config['actor_hiddens'] = [64, 64]
            config['actor_hidden_activation'] = 'relu'
            config['actor_lr'] = 0.00001  # in article 'ddpg'
            config['critic_lr'] = 0.00001
            config['critic_hiddens'] = [64, 64]
            config['critic_hidden_activation'] = 'relu'
            config['gamma'] = 0.99
            config['model']['fcnet_hiddens'] = [256, 256]
            config['model']['fcnet_activation'] = 'tanh'
            config['target_network_update_freq'] = 1 # change this
            config['lr'] = 1e-5
            # exploration
            config['exploration_config']['final_scale'] = 0.02
            config['exploration_config']['scale_timesteps'] = 1500000
            config['exploration_config']["initial_scale"] = 0.7
            config['exploration_config']["random_timesteps"] = 1000
            config['exploration_config']["stddev"] = 0.1
            del config['exploration_config']['ou_base_scale']
            del config['exploration_config']['ou_theta']
            del config['exploration_config']['ou_sigma']
            config['exploration_config']['type'] = 'GaussianNoise'
            # optimization
            config['tau'] = 0.005
            config['l2_reg'] = 5e-6 # critic loss
            config['train_batch_size'] = 256
            # each rollout worker has the train_batch_size = config['train_batch_size']/rollout_fragment_length
            config['rollout_fragment_length'] = 1
            config['learning_starts'] = 120000
            # evaluation
            config['timesteps_per_iteration'] = 3000
            # config['evaluation_interval'] = 5
            config['buffer_size'] = 300000 #3e5
            config["prioritized_replay_beta_annealing_timesteps"] = 2000000
            config['prioritized_replay'] = False
            config['final_prioritized_replay_beta'] = 0.4
            config['prioritized_replay_eps'] = 0.00001
            config['clip_rewards'] = False


        else:# merge
            config['n_step'] = 1
            config['actor_hiddens'] = [64, 64]
            config['actor_lr'] = 0.001  # in article 'ddpg'
            config['critic_lr'] = 0.0001
            config['critic_hiddens'] = [64, 64]
            config['gamma'] = 0.99
            config['model']['fcnet_hiddens'] = [64, 64]
            config['lr']=1e-5
            # exploration
            config['exploration_config']['final_scale'] = 0.05
            config['exploration_config']['scale_timesteps'] = 1500000
            config['exploration_config']['ou_base_scale'] = 0.1
            config['exploration_config']['ou_theta'] = 0.15
            config['exploration_config']['ou_sigma'] = 0.2
            # optimization
            config['tau'] = 0.002
            config['l2_reg'] = 1e-6
            config['train_batch_size'] = 64
            config['learning_starts'] = 3000
            # evaluation
            #config['evaluation_interval'] = 5
            config['buffer_size'] = 300000 #3e5
            config['timesteps_per_iteration'] = 3000
            config['prioritized_replay']=False

    
    #common config

    config['framework']='torch'
    config['callbacks'] = {
        "on_episode_end": None,
        "on_episode_start": None,
        "on_episode_step": None,
        "on_postprocess_traj": None,
        "on_sample_end": None,
        "on_train_result": None
    }  
    # config["opt_type"]= "adam" for impala and APPO, default is SGD
    # TrainOneStep class call SGD -->execution_plan function can have policy update function
    print("cuda is available: ", torch.cuda.is_available())
    print('Beginning training.')
    print("==========================================")
    print("running algorithm: ", alg_run)  # "Framework: ", "torch"

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # multiagent configuration
    if policy_graphs is not None:
        print("policy_graphs", policy_graphs)
        config['multiagent'].update({'policies': policy_graphs})
    if policy_mapping_fn is not None:
        config['multiagent'].update(
            {'policy_mapping_fn': tune.function(policy_mapping_fn)})
    if policies_to_train is not None:
        config['multiagent'].update({'policies_to_train': policies_to_train})

    create_env, gym_name = make_create_env(params=flow_params)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


def train_rllib(submodule, flags):
    """Train policies using the PPO algorithm in RLlib."""
    import ray
    from ray.tune import run_experiments
    start_time = timeit.default_timer()
    flow_params = submodule.flow_params
    print("the number of cpus: ", submodule.N_CPUS)
    n_cpus = submodule.N_CPUS
    n_rollouts = submodule.N_ROLLOUTS
    policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
    policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
    policies_to_train = getattr(submodule, "policies_to_train", None)

    alg_run, gym_name, config = setup_exps_rllib(
        flow_params, n_cpus, n_rollouts,
        policy_graphs, policy_mapping_fn, policies_to_train, flags)

    ray.init(num_cpus=n_cpus + 1, num_gpus=1,object_store_memory=200 * 1024 * 1024)
    # checkpoint and num steps setting
    if alg_run=="PPO":
        flags.num_steps = 1500
        checkpoint_freq = 100
    elif alg_run=="DDPG":
        flags.num_steps = 800
        checkpoint_freq = 40
    
    exp_config = {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": checkpoint_freq,
        "checkpoint_at_end": True,
        "max_failures": 999,
        "stop": {
            "training_iteration": flags.num_steps,
        },
    }
    print("training_iteration: ",exp_config["stop"]["training_iteration"])
    if flags.checkpoint_path is not None:
        exp_config['restore'] = flags.checkpoint_path
    print("=================Configs=================")
    for key in exp_config["config"].keys():
        if key == "env_config":  # you can check env_config in exp_configs directory.
            continue
        # no checking None or 0 value at all.
        # elif exp_config["config"][key] == None or exp_config["config"][key] == 0:
        #    continue
        elif key == "model":  # model checking
            print("----model config----")
            for key_model in exp_config["config"]["model"].keys():
                print(key_model, ":", exp_config["config"]["model"][key_model])
                # no checking None or 0 value at all.
                # if exp_config["config"][key] == None or exp_config["config"][key] == 0:
                #    continue
        else:
            print(key, ":", exp_config["config"][key])
    # change config data at the end of training (need to record time value to fix it)
    import time
    time.time()
    file_path_day=time.strftime('%Y-%m-%d', time.localtime(time.time()))
    file_path_hour=time.strftime('%H-%M-%S', time.localtime(time.time()))
    experiment_json='experiment_state-'+file_path_day+'_'+file_path_hour+'.json'
    # print experiment.json information
    print("=========================================")
    run_experiments({flow_params["exp_tag"]: exp_config})
    stop_time = timeit.default_timer()
    run_time = stop_time-start_time
    print("Training is Finished")
    print("total runtime: ", run_time)

    # modify params.json for testing that trained well
    saved_experiment_json_path=os.path.join("/home",getpass.getuser(),"ray_results",flow_params["exp_tag"],experiment_json)

    if os.path.exists(os.path.dirname(saved_experiment_json_path)) ==False:
        if int(experiment_json[-6]=="9"):
            experiment_json[-7]=str(int(experiment_json[-7])+1)
            experiment_json[-6]="0"
        else:
            experiment_json[-6]=str(int(experiment_json[-6])+1)
        saved_experiment_json_path=os.path.join("/home",getpass.getuser(),"ray_results",flow_params["exp_tag"],experiment_json)
    # check file is existed
    with open(saved_experiment_json_path,'r') as f:
        experiment_data=json.load(f)
        saved_params_json_path=os.path.join(experiment_data["checkpoints"][0]['logdir'],"params.json")
        print("params.json is located at : ",saved_params_json_path)
    #params.json open and modify value of exploration and ringlength for visualizing
    with open(saved_params_json_path,'r')as fin:
        params_data=json.load(fin)
    
    params_data['explore']=False
    paramStr=params_data["env_config"]["flow_params"]
    #fix ring length option
    if flags.exp_config=="singleagent_ring":
        paramStr=paramStr.replace("220","260")
        paramStr=paramStr.replace("270","260")

    with open(saved_params_json_path,'w')as fout:
        params_data["env_config"]["flow_params"]=paramStr
        json.dump(params_data,fout,indent="\t")
    print("Visualizing is Now Available")
    #Done

def main(args):
    """Perform the training operations."""
    # Parse script-level arguments (not including package arguments).
    flags = parse_args(args)

    # Import relevant information from the exp_config script.
    module = __import__(
        "exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    module_ma = __import__(
        "exp_configs.rl.multiagent", fromlist=[flags.exp_config])

    # rl part
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
        multiagent = False
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config.")

    # Perform the training operation.
    train_rllib(submodule, flags)


if __name__ == "__main__":
    main(sys.argv[1:])
