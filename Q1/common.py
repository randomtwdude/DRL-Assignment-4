# Common imports
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import itertools
import time

# Create Pendulum-v1 environment
def make_env_1():
    env = gym.make("Pendulum-v1", render_mode='rgb_array')
    return env

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

# Networks
LOG_STD_MIN = -22
LOG_STD_MAX = 2

# Xavier init
def weights_init_(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight, gain = 1)
        torch.nn.init.constant_(layer.bias, 0)


class FCFF(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in, 256),     nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
            nn.Linear(256, dim_out)
        )
        self.apply(weights_init_)


    def forward(self, x):
        return self.model(x)


class QNetwork(nn.Module):
    def __init__(self, dim_obs, action_space):
        super().__init__()
        dim_act = action_space.shape[0]
        self.layers1 = FCFF(dim_obs + dim_act, 1)
        self.layers2 = FCFF(dim_obs + dim_act, 1)


    def forward(self, obs, act):
        x = torch.cat([obs, act], dim = -1)
        return (
            self.layers1(x),
            self.layers2(x)
        )


class PNetwork(nn.Module):
    def __init__(self, dim_obs, action_space):
        super().__init__()
        dim_act = action_space.shape[0]
        self.layers = FCFF(dim_obs, dim_act * 2)

        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        self.action_bias  = torch.FloatTensor((action_space.high + action_space.low) / 2.)


    def forward(self, obs):
        output     = self.layers(obs)
        mean, lstd = torch.chunk(output, chunks = 2, dim = -1)

        lstd = torch.clamp(lstd, LOG_STD_MIN, LOG_STD_MAX)
        return mean, lstd


    def sample(self, obs):
        mean, lstd = self.forward(obs)
        std        = torch.exp(lstd)

        N   = torch.distributions.Normal(mean, std)
        x   = N.rsample()
        tx  = torch.tanh(x)
        act = tx * self.action_scale + self.action_bias

        log_prob = N.log_prob(x) - torch.log(self.action_scale * (1 - tx.pow(2)) + 1e-7)
        log_prob = log_prob.sum(1, keepdim = True)

        best = torch.tanh(mean) * self.action_scale + self.action_bias
        return act, log_prob, best