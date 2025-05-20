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
from collections import deque
from math import sqrt

from dmc import make_dmc_env


# Create Pendulum-v1 environment
def make_env_1():
    env = gym.make("Pendulum-v1", render_mode='rgb_array')
    return env


def make_env_2():
	# Create environment with state observations
	env_name = "cartpole-balance"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten = True, use_pixels = False)
	return env


def make_env_3():
    # Create environment with state observations
    env_name = "humanoid-walk"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten = True, use_pixels = False)
    return env


def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)


# Networks
LOG_STD_MIN = -20
LOG_STD_MAX = 2

HIDDEN_LAYER = 256

# Xavier init
def weights_init_(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain = 1)
        nn.init.constant_(layer.bias, 0)


class FCFF(nn.Module):
    def __init__(self, dim_in, dim_out, hidden = HIDDEN_LAYER, n_hidden = 1):
        super().__init__()

        layers = [nn.Linear(dim_in, hidden), nn.ReLU()]
        for _ in range(n_hidden):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]

        layers += [nn.Linear(hidden, dim_out)]
        self.model = nn.Sequential(*layers)
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
        return (self.layers1(x), self.layers2(x))


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

# ICM
class Curiosity(nn.Module):
    def __init__(self, dim_obs, action_space, hidden = HIDDEN_LAYER):
        super().__init__()
        dim_act = action_space.shape[0]

        self.encode = nn.Sequential(
            nn.Linear(dim_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU()
        )
        self.fw = nn.Sequential(
            nn.Linear(hidden + dim_act, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.rv = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU6(),
            nn.Linear(hidden, dim_act)
        )
        self.apply(weights_init_)

    def forward(self, state, next_state, act):
        rep      = self.encode(state)
        next_rep = self.encode(next_state)

        two_states = torch.cat([rep, next_rep], dim = -1)
        states_act = torch.cat([rep, act], dim = -1)

        return (
            self.rv(two_states),
            self.fw(states_act),
            next_rep
        )