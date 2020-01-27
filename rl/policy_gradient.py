import gym
import numpy as np
import torch

import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax
from collections import deque

from rl.agent import Agent
from rl.utils import Params, Utils

import matplotlib.pyplot as plt


# Followed tutorial/explanation from here: https://towardsdatascience.com/breaking-down-richard-suttons-policy-gradient-9768602cb63b

class PolicyGradientReinforceWithBaseline:
    def __init__(self):

        self.NUM_EPOCHS = Params.NUM_EPOCHS
        self.ALPHA = Params.ALPHA
        self.BATCH_SIZE = Params.BATCH_SIZE
        self.GAMMA = Params.GAMMA
        self.HIDDEN_SIZE = Params.HIDDEN_SIZE
        self.BETA = Params.BETA
        self.DEVICE = torch.device('cpu')

        self.rewards = []

        # create the environment
        self.env = gym.make('LunarLander-v2')

        self.agent = Agent(observation_space_size=self.env.observation_space.shape[0],
                           action_space_size=self.env.action_space.n,
                           hidden_size=self.HIDDEN_SIZE).to(self.DEVICE)

        self.adam = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)
        self.total_rewards = deque([], maxlen=100)

        self.finished_epoch = False

    def solve_environment(self):
        """
            Implementation of REINFORCE with baseline algorithm
        """
        episode = 0
        epoch = 0

        epoch_logs = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

        while True:

            # play an episode of the environment
            (weighted_log_prob_trajectory,
             episode_logits,
             sum_of_episode_rewards,
             episode) = self.play_episode(episode=episode)

            self.total_rewards.append(sum_of_episode_rewards)

            # get weighted log-probabilities of actions
            weighted_log_probs = torch.cat((weighted_log_probs, weighted_log_prob_trajectory), dim=0)
            epoch_logs = torch.cat((epoch_logs, episode_logits), dim=0)

            if episode >= self.BATCH_SIZE:
                self.finished_epoch = False
                episode = 0
                epoch += 1

                # calculate the loss and update agent
                loss, entropy = Utils.calculate_loss(epoch_logits=epoch_logs, weighted_log_probs=weighted_log_probs, beta=self.BETA)
                self.adam.zero_grad()
                loss.backward()
                self.adam.step()

                print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}")
                self.rewards.append(np.mean(self.total_rewards))

                # reset epoch arrays
                epoch_logs = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
                weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

                if np.mean(self.total_rewards) > 200:
                    print('Finished')
                    break

        self.env.close()

    def play_episode(self, episode: int):
        """
        Run through an episode
        """

        state = self.env.reset()

        # initialize the episode arrays
        episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)
        episode_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        average_rewards = np.empty(shape=(0,), dtype=np.float)
        episode_rewards = np.empty(shape=(0,), dtype=np.float)

        while True:

            if not self.finished_epoch:
                self.env.render()

            # get the action preferences from agent
            action_logits, episode_logits, action = self.agent.choose_action(state, episode_logits)

            episode_actions = torch.cat((episode_actions, action), dim=0)

            # observe state and reward
            state, reward, done, _ = self.env.step(action=action.cpu().item())

            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)
            average_rewards = np.concatenate((average_rewards, np.expand_dims(np.mean(episode_rewards), axis=0)), axis=0)

            if done:

                episode += 1

                discounted_rewards_to_go = Utils.get_discounted_rewards(rewards=episode_rewards, gamma=self.GAMMA)
                discounted_rewards_to_go -= average_rewards  # baseline

                mask = one_hot(episode_actions, num_classes=self.env.action_space.n)

                episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)
                episode_weighted_log_probs = episode_log_probs * \
                    torch.tensor(discounted_rewards_to_go).float().to(self.DEVICE)

                sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)
                sum_of_rewards = np.sum(episode_rewards)

                self.finished_epoch = True

                return sum_weighted_log_probs, episode_logits, sum_of_rewards, episode


def main():
    policy_gradient = PolicyGradientReinforceWithBaseline()
    policy_gradient.solve_environment()

    plt.plot(policy_gradient.rewards)
    plt.show()



if __name__ == "__main__":
    main()