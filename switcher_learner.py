import yaml
import datetime
import os
import pprint

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Recurrent, DataParallelNet
from tianshou.utils.net.discrete import Actor, Critic

from SwitcherEnv import make_env


class ConfigDict(dict):

  def __init__(self, config_dict):
    flat_config = self.flat(config_dict)
    super(ConfigDict, self).__init__(flat_config)
    self.__dict__ = self

  def __setattr__(self, __name: str, __value) -> None:
    return super().__setattr__(__name, __value)

  def flat(self, config):
    flat_config = dict()
    for k in config:
      if type(config[k]) == dict:
        # drop first level keys, promote second level keys to first level
        for sk, v in config[k].items():
          flat_config[sk] = v
      else:  # keep first level kv-pair unchanged
        flat_config[k] = config[k]

    return flat_config


with open("config.yaml", 'r') as f:
  em_config = yaml.safe_load(f)
  config = ConfigDict(em_config)


#*
def test_ppo(config):
  #*
  start_date = '2019-01-19'
  lstm_lag_steps = 10  # 10 units of frequency e.g., 1h, 12h.
  config.device = 'cpu'

  env, train_envs, test_envs = make_env(start_date, config.num_envs_per_worker)
  config.state_shape = env.observation_space.shape or env.observation_space.n
  config.action_shape = env.action_space.shape or env.action_space.n
  # seed
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)

  # model
  net_a = Recurrent(
      lstm_lag_steps,
      config.state_shape,
      config.action_shape,
      device=config.device).to(config.device)
  net_c = Recurrent(lstm_lag_steps, config.state_shape, config.action_shape,
                    config.device).to(config.device)
  if torch.cuda.is_available() and config.device == 'gpu':
    actor = DataParallelNet(
        Actor(
            net_a,
            config.action_shape,
            preprocess_net_output_dim=config.action_shape,
            device=None).to(config.device))
    critic = DataParallelNet(
        Critic(net_c, preprocess_net_output_dim=1,
               device=None).to(config.device))
  else:
    actor = Actor(
        net_a,
        config.action_shape,
        preprocess_net_output_dim=config.action_shape,
        device=config.device).to(config.device)
    critic = Critic(
        net_a, preprocess_net_output_dim=1,
        device=config.device).to(config.device)

  # NN Initialization
  for m in list(actor.modules()) + list(critic.modules()):
    if isinstance(m, torch.nn.Linear):
      # orthogonal initialization
      torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
      torch.nn.init.zeros_(m.bias)

  optim = torch.optim.Adam(
      list(actor.parameters()) + list(critic.parameters()), lr=config.lr)

  lr_scheduler = None
  if config.lr_decay:
    # decay learning rate to 0 linearly
    max_update_num = np.ceil(
        config.step_per_epoch / config.step_per_collect) * config.max_epoch

    lr_scheduler = LambdaLR(
        optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

  dist = torch.distributions.Categorical

  policy = PPOPolicy(
      actor,
      critic,
      optim,
      dist,
      discount_factor=config.gamma,
      gae_lambda=config.gae_lambda,
      max_grad_norm=config.max_grad_norm,
      vf_coef=config.vf_coef,
      ent_coef=config.ent_coef,
      reward_normalization=config.rew_norm,
      action_scaling=True,
      action_bound_method=config.bound_action_method,
      lr_scheduler=lr_scheduler,
      action_space=env.action_space,
      eps_clip=config.eps_clip,
      value_clip=config.value_clip,
      dual_clip=config.dual_clip,
      advantage_normalization=config.norm_adv,
      recompute_advantage=config.recompute_adv,
  )

  # load a previous policy
  if config.resume_path:
    ckpt = torch.load(config.resume_path, map_location=config.device)
    policy.load_state_dict(ckpt["model"])
    train_envs.set_obs_rms(ckpt["obs_rms"])
    test_envs.set_obs_rms(ckpt["obs_rms"])
    print("Loaded agent from: ", config.resume_path)

  # collector
  if config.num_envs_per_worker > 1:
    buffer = VectorReplayBuffer(
        config.buffer_size, len(train_envs), stack_num=lstm_lag_steps)
  else:
    buffer = ReplayBuffer(config.buffer_size, stack_num=lstm_lag_steps)
  train_collector = Collector(
      policy, train_envs, buffer, exploration_noise=True)
  test_collector = Collector(policy, test_envs)

  # log
  now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
  config.algo_name = "ppo"
  log_name = os.path.join(config.env, config.algo_name, str(config.seed), now)
  log_path = os.path.join(config.logdir, log_name)

  # logger
  writer = SummaryWriter(log_path)
  writer.add_text("config", str(config))
  if config.logger == "wandb":
    logger = WandbLogger(
        save_interval=1,
        name=log_name.replace(os.path.sep, "__"),
        run_id=config.resume_id,
        config=config,
        project=config.wandb_project,
    )
    logger.load(writer)
  elif config.logger == "tensorboard":
    logger = TensorboardLogger(writer)

  def save_best_fn(policy):
    state = {"model": policy.state_dict(), "obs_rms": train_envs.get_obs_rms()}
    torch.save(state, os.path.join(log_path, "policy.pth"))

  # trainer
  result = onpolicy_trainer(
      policy,
      train_collector,
      test_collector,
      config.max_epoch,
      config.step_per_epoch,
      config.repeat_per_collect,
      config.eval_num_envs_per_worker,
      config.minibatch_size,
      step_per_collect=config.step_per_collect,
      save_best_fn=save_best_fn,
      logger=logger,
      test_in_train=False,
  )
  pprint.pprint(result)


#*
# if __name__ == "__main__":
test_ppo(config)
# # Let's watch its performance!
# if not config.watch:
#   policy.eval()
#   test_envs.seed(config.seed)
#   test_collector.reset()
#   result = test_collector.collect(
#       n_episode=config.eval_num_envs_per_worker, render=config.render)
#   print(
#       f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
