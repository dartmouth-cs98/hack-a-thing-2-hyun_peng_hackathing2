import numpy as np
import torch
from torch.nn.functional import softmax, log_softmax

class Params:
    NUM_EPOCHS = 1000
    ALPHA = 1e-3        # learning rate
    BATCH_SIZE = 32     # how many episodes we want to pack into an epoch
    GAMMA = 0.99        # discount rate
    HIDDEN_SIZE = 128    # number of hidden nodes we have in our dnn
    BETA = 0.1          # the entropy bonus multiplier


class Utils:
    @staticmethod
    def get_discounted_rewards(rewards: np.array, gamma: float) -> np.array:
        discounted_rewards = np.empty_like(rewards, dtype=np.float)
        for i in range(rewards.shape[0]):
            gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_gammas)
            discounted_rewards[i] = discounted_reward
        return discounted_rewards

    @staticmethod
    def calculate_loss(epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor, beta) -> (torch.Tensor, torch.Tensor):
        policy_loss = -1 * torch.mean(weighted_log_probs)

        # add the entropy bonus
        p = softmax(epoch_logits, dim=1)
        log_p = log_softmax(epoch_logits, dim=1)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
        entropy_bonus = -1 * beta * entropy

        return policy_loss + entropy_bonus, entropy