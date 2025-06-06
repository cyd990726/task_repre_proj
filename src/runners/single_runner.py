from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class SingleRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = 1 # 这是batch_size_run

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.test_returns = []


    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self):
        """ When in pretrain_phase, we choose action randomly and don't do recording.
        """
        for _ in range(10):
            self.reset()

            terminated = False
            episode_return = 0
            self.mac.init_hidden(batch_size=self.batch_size)

            while not terminated:

                pre_transition_data = {
                    "state": [self.env.get_state()],
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [self.env.get_obs()]
                }

                self.batch.update(pre_transition_data, ts=self.t)

                # Pass the entire batch of experiences up till now to the agents
                # Receive the actions for each agent at this timestep in a batch of size 1
                
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True)

                reward, terminated, env_info = self.env.step(actions[0])
                episode_return += reward

                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }

                self.batch.update(post_transition_data, ts=self.t)

                self.t += 1

            last_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(last_data, ts=self.t)

            # Select actions in the last stored state
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True)
            
            self.batch.update({"actions": actions}, ts=self.t)
            self.test_returns.append(episode_return)
        
        # 返回10个episode的平均return
        return np.mean(self.test_returns)

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
