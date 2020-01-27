
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.distributions import Categorical

class Agent(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Agent, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)
        )

    def forward(self, x):
        x = normalize(x, dim=1)
        x = self.net(x)
        return x

    def choose_action(self, state, episode_logits, device=torch.device('cpu')):
        action_logits = self(torch.tensor(state).float().unsqueeze(dim=0).to(device))
        episode_logits = torch.cat((episode_logits, action_logits), dim=0)
        action = Categorical(logits=action_logits).sample()

        return action_logits, episode_logits, action