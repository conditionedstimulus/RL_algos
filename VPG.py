import argparse
import logging

import gym
import numpy as np

import torch as T
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical

class PN(nn.Module): #  Policy network

    def __init__(self, n_inputs, n_actions, n_hidden):
        super(PN, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )


    def forward(self, state):
        norm_state = self._format(state)
        logits = self.actor(norm_state)
        probs = F.softmax(logits, dim=1)

        return probs


    @staticmethod
    def _format(state):
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32)
            state = state.unsqueeze(0)
        return state


    def get_action(self, state):
        # Calc the probabilities
        probs = self.forward(state)

        m = Categorical(probs)
        action = m.sample()
        logprobs = m.log_prob(action).unsqueeze(-1)
        return action.item(), logprobs



class SVN(nn.Module): # State-value network

    def __init__(self, n_inputs, n_hidden, n_actions=1):
        super(SVN, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )


    def forward(self, state):
        norm_state = self._format(state)
        state_value = self.critic(norm_state)

        return state_value

    
    @staticmethod
    def _format(state):
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32)
            state = state.unsqueeze(0)
        return state
        


class Agent:
    
    def __init__(self, env_name, n_hidden, gamma, seed):
        
        self.env        = gym.make(env_name)
        self.gamma      = gamma
        self.seed       = seed
        self.n_actions  = self.env.action_space.n
        self.n_obs      = self.env.observation_space.shape[0]
        self.agent      = PN(self.n_obs, self.n_actions, n_hidden)
        self.value_n    = SVN(self.n_obs, n_hidden)
        self.opti_ag    = T.optim.Adam(self.agent.parameters(), 0.001)
        self.opti_v     = T.optim.RMSprop(self.value_n.parameters(), 0.001)


    def train(self, N_EPISODES):

        self.env.seed(self.seed)
        returns_episode = list()

        for e in range(N_EPISODES):
            state           = self.env.reset()
            log_probs       = list()
            rewards         = list()
            state_values    = list()

            for _ in range(10000):
                
                action, log_p = self.agent.get_action(state)
                new_state, reward, done, _ = self.env.step(action)

                # Calculate state value
                state_v = self.value_n(state)
                
                state_values.append(state_v)
                log_probs.append(log_p)
                rewards.append(reward)

                if done:
                    #  Append the sum of the returns per episode
                    returns_episode.append(np.sum(rewards))
                    break

                state = new_state

            # Optimizing the agent
            self.optimization(rewards, log_probs, state_values)

            if e % 100 == 0:
                # Write out the mean value of the rewards every 100 episodes
                logging.info(f'Episode: {e} the mean value of the last 100 episodes: {np.mean(returns_episode[-100:])}')


    def optimization(self, returns, log_probs_,  state_v):

        # calculating the discounted cumulative returns
        cumulative_r = self.calc_returns(returns, self.gamma)

        # Transform lists to tensors
        cumulative_r = T.FloatTensor(cumulative_r).unsqueeze(1)
        state_v = T.cat(state_v)

        #  Calculate the value error
        value_err = cumulative_r - state_v

        # calculating the loss function
        loss = -T.cat([lp_ * v for lp_, v in zip(log_probs_, value_err.detach())]).sum()

        # PN steps
        self.opti_ag.zero_grad()
        # backpropagation
        loss.backward()
        self.opti_ag.step()

        # SVN steps
        self.opti_v.zero_grad()
        value_loss = (value_err**2).sum()
        # backpropagation
        value_loss.backward()
        self.opti_v.step()


    @staticmethod
    def calc_returns(rewards, gamma):
    
        returns = list()
        cumulated_reward = 0

        for r_t in rewards[::-1]:
            cumulated_reward = r_t + (cumulated_reward * gamma)
            returns.append(cumulated_reward)
        
        return returns[::-1]




def main():
        
    parser = argparse.ArgumentParser(description='Vanilla Policy Gradient method (VPG)')
    parser.add_argument('--env_name', choices=['CartPole-v0'], default='CartPole-v0', help="The name of the OpenAI' environment")
    parser.add_argument('--n_hidden', type=int, default=128, metavar='N', help='The number of nodes in the single hidden layer')
    parser.add_argument('--n_eps', type=int, default=2000, metavar='N', help='The number of episodes')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor')
    parser.add_argument('--seed', type=int, default=92, metavar='N', help='random seed')
    args = parser.parse_args()

    T.manual_seed(args.seed) 

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])

    agent = Agent(args.env_name, args.n_hidden, args.gamma, args.seed)
    logging.info(f'Agent has created... Environment: {args.env_name}')

    logging.info('Training has started...')
    agent.train(args.n_eps)

    logging.info('End of the program...')




if __name__ == '__main__':
    main()