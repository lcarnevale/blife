import gym
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

env = gym.make('gympy:blife-v0')
torch.manual_seed(1)

# Hyperparameters
learning_rate = 0.01
gamma = 0.99


class Policy(nn.Module):
    def __init__(self, observation_space_size, action_space_size):
        super(Policy, self).__init__()        

        self.l1 = nn.Linear(observation_space_size, 128, bias=False)
        self.l2 = nn.Linear(128, action_space_size, bias=False)
        self.gamma = gamma
        
        # Episode policy and reward history 
        self.policy_history = torch.empty((), dtype=torch.float32, device = 'cpu')
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)

policy = Policy(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

def select_action(state):
    #Select an action by running policy model and
    # choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = Categorical(state)
    action = c.sample()
 
    # Add log probability of our chosen action to our history    
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))
    return action

def update_policy():
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = torch.empty((), dtype=torch.float32, device = 'cpu')
    policy.reward_episode= []

def main(episodes=30, MAX_STEP_NUM=30):
    running_reward = 0
    for episode in range(episodes):
        # Reset environment and record the starting state
        state = env.reset()
        done = False
    
        for step in range(1, MAX_STEP_NUM+1):
            action = select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.item())

            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break
        reward_episode = np.sum(policy.reward_episode)
        
        # Used to determine when the environment is solved.
        running_reward = (running_reward * gamma) + (step * learning_rate)

        update_policy()
        print('Episode %d\tLast length (steps): %d\tAverage length (steps): %.2f\tReward episode: %.3f\tLoss: %s' % (episode, step, running_reward, reward_episode, policy.loss_history[-1]))

        env.close()


        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and the last episode runs to {} step steps!".format(running_reward, step))
        #     break
    
    # window = int(episodes/1)
    window = 1

    def moving_average(data, window):
        return np.convolve(data, np.ones(window), 'valid') / window

    # fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9,9])

    fig, ((ax1)) = plt.subplots(1, 1, figsize=[9,9])
    rolling_mean = moving_average(np.array(policy.reward_history), window)
    std = np.array(policy.reward_history).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(policy.reward_history)),rolling_mean-std, rolling_mean+std, color='orange', alpha=0.2)
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')
    fig.tight_layout(pad=2)
    fig.savefig('mean_average.png')

    fig, ((ax2)) = plt.subplots(1, 1, figsize=[9,9])
    ax2.plot(policy.reward_history)
    # ax2.set_title('Reward')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    fig.tight_layout(pad=2)
    fig.savefig('reward.png')

    fig, ((ax3)) = plt.subplots(1, 1, figsize=[9,9])
    ax3.plot(policy.loss_history)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    fig.tight_layout(pad=2)
    fig.savefig('loss.png')


if __name__ == "__main__":
    main()