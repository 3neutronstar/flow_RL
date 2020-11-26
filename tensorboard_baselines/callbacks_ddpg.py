import numpy as np

from stable_baselines3 import ddpg
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose,self.model)
    
    def init_callback(self,model):
        super(TensorboardCallback,self).init_callback(model=self.model)
        self.model=model
    def on_step(self) -> bool:
        # Log scalar value (here a random variable)
        episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )
        self.logger.record('epi/reward', episode_rewards)
        self.n_calls += 1
        # timesteps start at zero
        self.num_timesteps = self.model.num_timesteps + 1

        return self._on_step()