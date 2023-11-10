import time
from matplotlib import cm, pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.algorithms.policy_search import GPOMDP
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gym
from mushroom_rl.policy import StateStdGaussianPolicy
from mushroom_rl.utils.dataset import compute_J, compute_metrics
from mushroom_rl.utils.optimizers import *

from tqdm import tqdm, trange

from Discord import send_msg_with_file_to_discord_channel
url = 'https://discord.com/api/webhooks/1169236452987637800/x3p9keSTZh9le91Dqb5y9QoeHejsQ5jzrDNyrnmhjr50I8tEb8-5d1wNs2U4PZlMpZ86' # playground
url = 'https://discord.com/api/webhooks/1105492131457544294/8QBRJc1N5m1pa0pM64INmaDUl6J1DNg3t8XhV5u936P99m5RL_rNiFNJe6cZ-C6ITdSO' # mac-book-pro
import inspect

tqdm.monitor_interval = 0

DISCORD_FREQUENCY = 10
MEAN_RETURNS = []
MEAN_STDS = []

class MuNetwork(nn.Module):
    # This network will predict the mean of the action distribution
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(MuNetwork, self).__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._mu = nn.Linear(n_features, n_output)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self._h1.weight)
        nn.init.xavier_uniform_(self._mu.weight)

    def forward(self, state, **kwargs):
        features = torch.tanh(self._h1(state))
        mu = self._mu(features)
        return mu

class SigmaNetwork(nn.Module):
    # This network will predict the standard deviation of the action distribution
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(SigmaNetwork, self).__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._sigma = nn.Linear(n_features, n_output)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self._h1.weight)
        nn.init.xavier_uniform_(self._sigma.weight)

    def forward(self, state, **kwargs):
        features = torch.tanh(self._h1(state))
        # Use softplus to ensure the output is positive
        sigma = F.softplus(self._sigma(features))
        return sigma

def send_mean_returns_to_discord(epoche, MEAN_RETURNS):
    plt.figure(figsize=(10, 5))
    plt.plot(MEAN_RETURNS, color='green')
    plt.title(f'GPOMDP on Pendulum-v1 Mean Returns at Epoche {epoche}')
    plt.xlabel('epoche')
    plt.ylabel('Mean Returns')
    plt.savefig('Pendulum/Mushroom/MeanReturns/Mean_Returns_' + str(epoche) + '.png')
    plt.close()
    send_msg_with_file_to_discord_channel(content='', file=open('Pendulum/Mushroom/MeanReturns/Mean_Returns_' + str(epoche) + '.png', 'rb'), filename='Pendulum/Mushroom/MeanReturns/Mean_Returns_' + str(epoche) + '.png', url=url)

def send_mean_stds_to_discord(epoche, MEAN_STDS):
    plt.figure(figsize=(10, 5))
    plt.plot(MEAN_STDS, color='blue')
    plt.title(f'GPOMDP on Pendulum-v1 Mean $\sigma$ at Epoche {epoche}')
    plt.xlabel('epoche')
    plt.ylabel('Mean Standard Deviation')
    plt.savefig('Pendulum/Mushroom/MeanStd/MeanStd' + str(epoche) + '.png')
    plt.close()
    send_msg_with_file_to_discord_channel(content='', file=open('Pendulum/Mushroom/MeanStd/MeanStd' + str(epoche) + '.png', 'rb'), filename='Pendulum/Mushroom/MeanStd/MeanStd' + str(epoche) + '.png', url=url)

    
def experiment(n_epochs, n_iterations, ep_per_run):
    np.random.seed(1)

    logger = Logger(GPOMDP.__name__, results_dir='./Pendulum/Mushroom')
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + GPOMDP.__name__)

    # MDP
    mdp = Gym('Pendulum-v1', gamma=1.0, horizon=200)
    mdp.seed(1)

    # Create the approximators for mean and sigma using the new networks
    mu_approximator = Regressor(TorchApproximator,
                                                        input_shape=mdp.info.observation_space.shape,
                                                        output_shape=mdp.info.action_space.shape,
                                                        network=MuNetwork,
                                                        optimizer={'class': optim.Adam,
                                                                           'params': {'lr': 7e-4,
                                                                                            'eps': 1e-5
                                                                                            }
                                                                            },
                                                        n_features=32)

    sigma_approximator = Regressor(TorchApproximator,
                                                            input_shape=mdp.info.observation_space.shape,
                                                            output_shape=mdp.info.action_space.shape,
                                                            network=SigmaNetwork,
                                                            optimizer={'class': optim.Adam,
                                                                               'params': {'lr': 7e-4,
                                                                                                 'eps': 1e-5}
                                                                               },
                                                            n_features=32)

    policy = StateStdGaussianPolicy(mu_approximator, sigma_approximator)

    # Agent
    optimizer = AdamOptimizer(eps=1e-2)
    agentorithm_params = dict(optimizer=optimizer)
    agent = GPOMDP(mdp.info, policy, **agentorithm_params)

    # Train
    core = Core(agent, mdp)
    
    # Initial evaluation
    core.learn(n_episodes=n_iterations * ep_per_run, # number of episodes to move the agent;
                      n_episodes_per_fit= n_iterations * ep_per_run, # number of episodes between each fit of the policy;
                    )
    dataset_eval, env_info = core.evaluate(n_episodes=ep_per_run, get_env_info=True) # Plays ep_per_run episodes -> ep_per_run (25) * horizon steps (200)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma) # Discounted return
    R = compute_J(dataset_eval) # Undiscounted return
    std = env_info['std']
    logger.epoch_info(0, J=np.mean(J), R=np.mean(R), Std = np.mean(std)) # At this point, ep_per_run (25) * horizon steps (200) steps have been played = 5000 steps
    MEAN_RETURNS.append(np.mean(R))
    MEAN_STDS.append(np.mean(std))
    
    # Learning loop
    for i in trange(n_epochs, leave=False):
        start_time = time.time()
        core.learn(n_episodes=n_iterations * ep_per_run, # number of episodes to move the agent;
                          n_episodes_per_fit=n_iterations * ep_per_run # number of episodes between each fit of the policy;
                          )
        dataset_eval, env_info = core.evaluate(n_episodes=ep_per_run, get_env_info=True) # Plays ep_per_run episodes -> ep_per_run (25) * horizon steps (200)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        R = compute_J(dataset_eval)
        std = env_info['std']
        MEAN_RETURNS.append(np.mean(R))
        MEAN_STDS.append(np.mean(std))
        end_time = time.time()
        logger.epoch_info(i+1, J=np.mean(J), R=np.mean(R), Std = np.mean(std), Time=end_time-start_time)

        if i % DISCORD_FREQUENCY == 0:
            send_mean_returns_to_discord(i+1, MEAN_RETURNS)
            send_mean_stds_to_discord(i+1, MEAN_STDS)

if __name__ == '__main__':
    experiment(n_epochs=10000, n_iterations=1, ep_per_run=30)
        
# # https://github.com/MushroomRL/mushroom-rl/blob/95b5297733185183f1e7709d2d51f39de0665ae7/examples/lqr_pg.py#L67

