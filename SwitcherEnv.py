import random
import math
import time
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tianshou.env import DummyVectorEnv, VectorEnvNormObs

NUM_FEATURES = 5  # change this number to how many columns other than 'timestamp' and 'reward'
client = MongoClient("mongodb://localhost:27017/")
db = client["sample_data"]

class DataLoader:

  def __init__(self, start_date, num):
    self.candle_collection = db["candles"]
    self.num = num
    self.buffer = []
    self.index = 0

    if not start_date:
      earliest_timestamp = self.candle_collection.find_one(
          {},
          projection={
              "_id": 0,
              "timestamp": 1
          },
          sort=[("timestamp", pymongo.ASCENDING)])["timestamp"]
      latest_timestamp = self.candle_collection.find_one(
          {},
          projection={
              "_id": 0,
              "timestamp": 1
          },
          sort=[("timestamp", pymongo.DESCENDING)
               ])["timestamp"] - timedelta(hours=self.num)
      random_fraction = random.random()
      delta = (latest_timestamp - earliest_timestamp) * random_fraction
      self.start_date = earliest_timestamp + timedelta(
          seconds=delta.total_seconds())
    else:
      self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
    self.load_data()

  def load_data(self):
    cursor = self.candle_collection.find(
        {"timestamp": {
            "$gte": self.start_date
        }},
        projection={"_id": 0},
        limit=self.num,
        sort=[("timestamp", pymongo.ASCENDING)])
    self.buffer = list(cursor)
    if self.buffer:
      self.start_date = self.buffer[-1]["timestamp"]

  def __iter__(self):
    return self

  def __next__(self):
    if self.index >= len(self.buffer):
      self.load_data()
      self.index = 0

      if not self.buffer:
        return None, None, None

    row = self.buffer[self.index]
    timestamp = row["timestamp"]
    features = [
        value for key, value in row.items()
        if key not in ['timestamp', 'reward']
    ]
    reward = row["reward"]
    self.index += 1

    return timestamp, features, reward


class CustomEnv(gym.Env):

  def __init__(self, start_date=None, num=10000):
    super(CustomEnv, self).__init__()
    self.num_features = NUM_FEATURES

    self.start_date = start_date
    self.num = num
    self.data_loader = DataLoader(start_date, num)
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32)

  def reset(self):
    self.data_loader = DataLoader(self.start_date, self.num)
    timestamp, features, reward = next(iter(self.data_loader))
    if not reward:
      import pdb
      pdb.set_trace()
    return features, {}

  def step(self, action):
    timestamp, features, reward = next(iter(self.data_loader))

    if timestamp is None or features is None or reward is None:
      done = True
      return [0 for i in range(self.num_features)], 0, done, {}

    done = False
    return features, reward if action == 1 else 0, done, False, {}


def make_env(start_date='', training_num=4):
  """Wrapper function for Mujoco env.
    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.
    :return: a tuple of (single env, training envs, test envs).
  start_date=''
  training_num=4
  """
  env = CustomEnv(start_date)
  train_envs = DummyVectorEnv(
      [lambda: CustomEnv(start_date) for _ in range(training_num)])
  test_envs = DummyVectorEnv([lambda: CustomEnv(start_date) for _ in range(1)])
  # obs norm wrapper
  train_envs = VectorEnvNormObs(train_envs)
  test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
  test_envs.set_obs_rms(train_envs.get_obs_rms())
  return env, train_envs, test_envs

if __name__ == '__main__':
  # Usage example
  env = CustomEnv("", 100)
  obs = env.reset()
  for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    if done:
      break
