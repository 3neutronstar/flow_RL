"""Runner script for single and multi-agent reinforcement learning experiments.
This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.
Usage
    python train.py EXP_CONFIG
"""
import argparse
import json
import os
import sys
from time import strftime
from copy import deepcopy
import timeit
from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env


def parse_args(args):
    """Parse training options user can specify in command line.
    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')
    
    parser.add_argument(  # for stable-baselines3
        '--algorithm', type=str, default="DDPG",
    )  # choose algorithm in order to use
    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=400,
        help='How many total steps to perform learning over')
    parser.add_argument(
        '--rollout_size', type=int, default=600,
        help='How many steps are in a training batch.')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory with checkpoint to restore training from.')

    return parser.parse_known_args(args)[0]




def run_model_stablebaseline(flow_params,
                             num_cpus=1,
                             rollout_size=50,
                             num_steps=50,
                             algorithm="ppo",
                             exp_config=None
                             ):
    """Run the model for num_steps if provided.
    Parameters
    ----------
    flow_params : dict
        flow-specific parameters
    num_cpus : int
        number of CPUs used during training
    rollout_size : int
        length of a single rollout
    num_steps : int
        total number of training steps
    The total rollout length is rollout_size.
    Returns
    -------
    stable_baselines.*
        the trained model
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if num_cpus == 1:
        constructor = env_constructor(params=flow_params, version=0)()
        # The algorithms require a vectorized environment to run
        env = DummyVecEnv([lambda: constructor])
    else:
        env = SubprocVecEnv([env_constructor(params=flow_params, version=i)
                             for i in range(num_cpus)])
    if algorithm=="PPO":
        from stable_baselines3 import PPO
        train_model = PPO('MlpPolicy', env, verbose=1, n_steps=rollout_size)
        train_model.learn(total_timesteps=num_steps)
        print("Learning Process is Done.")
        return train_model

    elif algorithm =="DDPG":
        from stable_baselines3 import DDPG
        from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
        import numpy as np
        if exp_config=='singleagent_figure_eight':
            train_model = DDPG('MlpPolicy', env, verbose=1,
                                n_episodes_rollout=rollout_size,
                                learning_starts=3000,
                                learning_rate=0.0001,
                                action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(1),sigma=0.15*np.ones(1),initial_noise=0.7*np.ones(1)),
                                tau=0.005,
                                batch_size=128,
                                tensorboard_log='tensorboard_ddpg',
                                device='cuda',
                                )
        else:
            train_model = DDPG('MlpPolicy', env, verbose=1,
                            n_episodes_rollout=rollout_size,
                            learning_starts=1200,
                            tensorboard_log='tensorboard_ddpg',
                            learning_rate=0.0001,
                            action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(1),sigma=0.15*np.ones(1),initial_noise=0.7*np.ones(1)),
                            tau=0.005,
                            batch_size=512,
                            device='cpu',
                            )
        

        from tensorboard_baselines.callbacks_ddpg import TensorboardCallback
        train_model.learn(total_timesteps=num_steps,
                          log_interval=2,eval_log_path='ddpg_log',eval_freq=2, 
                          eval_freq=10,
                          #callback=[TensorboardCallback],
                          )
        print("Learning Process is Done.")
        return train_model

def train_stable_baselines(submodule, flags):
    """Train policies using the PPO algorithm in stable-baselines."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    flow_params = submodule.flow_params
    # Path to the saved files
    exp_tag = flow_params['exp_tag']
    result_name = '{}/{}'.format(exp_tag, strftime("%Y-%m-%d-%H:%M:%S"))

    # Perform training.
    start_time = timeit.default_timer()
    # print experiment.json information
    print("=========================================")
    print('Beginning training.')
    print('Algorithm :', flags.algorithm)
    model = run_model_stablebaseline(
        flow_params, flags.num_cpus, flags.rollout_size, flags.num_steps,flags.algorithm,flags.exp_config)

    stop_time = timeit.default_timer()
    run_time = stop_time-start_time
    print("Training is Finished")
    print("total runtime: ", run_time)
    # Save the model to a desired folder and then delete it to demonstrate
    # loading.
    print('Saving the trained model!')
    path = os.path.realpath(os.path.expanduser('~/baseline_results'))
    ensure_dir(path)
    save_path = os.path.join(path, result_name)
    model.save(save_path)

    # dump the flow params
    with open(os.path.join(path, result_name) + '.json', 'w') as outfile:
        json.dump(flow_params, outfile,
                  cls=FlowParamsEncoder, sort_keys=True, indent=4)

    # Replay the result by loading the model
    print('Loading the trained model and testing it out!')
    if flags.exp_config.lower()=="ppo":
        from stable_baselines3 import PPO
        model = PPO.load(save_path)
    elif flags.exp_config.lower()=="ddpg":
        from stable_baselines3 import DDPG
        model = DDPG.load(save_path)
    flow_params = get_flow_params(os.path.join(path, result_name) + '.json')
    flow_params['sim'].render = True
    env = env_constructor(params=flow_params, version=0)()
    # The algorithms require a vectorized environment to run
    eval_env = DummyVecEnv([lambda: env])
    obs = eval_env.reset()
    reward = 0
    for _ in range(flow_params['env'].horizon):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action)
        reward += rewards
    print('the final reward is {}'.format(reward))


def main(args):
    """Perform the training operations."""
    # Parse script-level arguments (not including package arguments).
    flags = parse_args(args)

    # Import relevant information from the exp_config script.
    module = __import__(
        "exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    module_ma = __import__(
        "exp_configs.rl.multiagent", fromlist=[flags.exp_config])

    # Import the sub-module containing the specified exp_config and determine
    # whether the environment is single agent or multi-agent.
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
        multiagent = False
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config.")

    # Perform the training operation.
    train_stable_baselines(submodule, flags)


if __name__ == "__main__":
    main(sys.argv[1:])
