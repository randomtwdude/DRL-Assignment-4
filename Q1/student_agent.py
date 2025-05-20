import gymnasium as gym
import numpy as np

from common import PNetwork, torch

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)

        self.policy = PNetwork(3, self.action_space)
        self.load("save_q1_250.bin")

    def act(self, observation):
        obs = torch.FloatTensor(observation).to("cpu").unsqueeze(0)
        _, _, act = self.policy.sample(obs)
        return act.detach().cpu().numpy()[0]

    def load(self, path):
        save = torch.load(path)
        self.policy.load_state_dict(save['policy_state_dict'])
        self.policy.eval()