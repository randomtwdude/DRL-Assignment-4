import gymnasium as gym
import numpy as np

from common import PNetwork, torch

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)

        self.policy = PNetwork(67, self.action_space)
        self.load("save_q3_1200.bin")

    def act(self, observation):
        obs = torch.FloatTensor(observation).to("cpu").unsqueeze(0)
        _, _, act = self.policy.sample(obs)
        return act.detach().cpu().numpy()[0]

    def load(self, path):
        save = torch.load(path)
        self.policy.load_state_dict(save['policy_state_dict'])
        self.policy.eval()
