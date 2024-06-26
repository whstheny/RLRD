import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn as nn
from copy import deepcopy
import random
from tqdm import tqdm
import math

import torch.nn as nn
import torch.nn.functional as F

epsilon = 1e-8

class PolicyNet(nn.Module):
    def __init__(self, num_teacher, embedding_length,device):
        super(PolicyNet, self).__init__()
        self.num_teacher = num_teacher

        # 定义权重和偏置
        self.W1 = nn.Parameter(torch.FloatTensor(embedding_length, 128).uniform_(-0.5, 0.5)) #768,128
        self.W2 = nn.Parameter(torch.FloatTensor(128, 128).uniform_(-0.5, 0.5)) #128,128
        self.W3 = nn.Parameter(torch.FloatTensor(num_teacher, 128).uniform_(-0.5, 0.5)) #2,128
        self.b = nn.Parameter(torch.FloatTensor(1, 128).uniform_(-0.5, 0.5))

        self.fc_alpha = nn.Parameter(torch.FloatTensor(128, 1).uniform_(-0.5, 0.5))
        self.fc_beta = nn.Parameter(torch.FloatTensor(128, 1).uniform_(-0.5, 0.5))

        self.device = device
    def forward(self, x1, x2, x3):
        # x1: (num_sent, batch_size, embedding_length) 2,128,768
        # x2: (num_teacher, batch_size, batch_size) 2,128,128
        # x3: (num_teacher, 1) 1,2

        x1_ = torch.matmul(x1, self.W1)  #2,128,128
        x2_ = torch.matmul(x2, self.W2)  #2,128,128

        x3_ = torch.matmul(x3, self.W3)  #1,128

        scaled_out = torch.relu(x1_ + x2_ + x3_ + self.b)
        # scaled_out = torch.clamp(scaled_out, min=1e-5, max=10 - 1e-5) #2,128,128
        scaled_out_reshaped = scaled_out.view(-1, 128)
        alpha = torch.matmul(scaled_out_reshaped, self.fc_alpha)
        beta = torch.matmul(scaled_out_reshaped, self.fc_beta)
        alpha = F.softplus(alpha).mean() + 50
        #alpha = torch.relu(alpha).mean() + 50
        beta = F.softplus(beta).mean() + 50
        #beta = torch.relu(beta).mean() + 50
        weights = [alpha, beta]

        return weights

    def take_action(self, state):

        weights = self.forward(*state)

        # dist = torch.distributions.Normal(weights[0].float(), weights[1].float())
        dist = torch.distributions.beta.Beta(weights[0].float(), weights[1].float())
        action = dist.sample().to(self.device)

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return action, weights

    def test_policy(self, state):
        avg_probability = self.forward(*state).to(self.device)
        action = torch.distributions.Bernoulli(avg_probability).sample().to(self.device)
        return action, avg_probability


from collections import namedtuple
import random
import torch
from torch.distributions import Bernoulli

probs = torch.tensor([0.5], requires_grad=True)

Transition = namedtuple('Transion',
                        ('state', 'action', 'weights', 'reward', 'value'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return self.memory

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = []
        self.position = 0


def optimize_model(memory, policy_net,critic, device, lr=1e-4):
    CLIP_EPSILON = 0.2
    NUM_PPO_UPDATES = 3
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    BATCH_SIZE = 10
    gamma = 0.99
    gae_lambda = 0.95
    num_batches = len(memory) // BATCH_SIZE
    all_transitions = memory.sample()
    batch = Transition(*zip(*all_transitions))
    for _ in range(NUM_PPO_UPDATES):

        # Prepare data
        action_batch = torch.cat(list(map(lambda a: torch.tensor([a], device=device), batch.action)))
        reward_batch = torch.cat(list(map(lambda r: torch.tensor([r], device=device), batch.reward)))
        old_weights = torch.cat(list(map(lambda r: torch.tensor(r, device=device), batch.weights)))
        old_weights = old_weights.view(-1, 2)

        value = torch.cat([torch.tensor([v], device=device) for v in batch.value])

        advantage = torch.zeros(len(reward_batch), dtype=torch.float32, device=device)
        for t in range(len(reward_batch) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_batch) - 1):
                a_t += 0.99 * (reward_batch[k] + gamma * value[k + 1] * - value[k])
            advantage[t] = a_t

        weights_list = []
        for state in batch.state:
            if isinstance(state, torch.Tensor):
                state = state.half().to(device)
            elif isinstance(state, list):
                state = [s.float().to(device) for s in state]
            weight = policy_net(*state)
            weights1 = torch.stack([weight[0], weight[1]]).unsqueeze(0)
            weights_list.append(weights1)
        weights = torch.cat(weights_list, dim=0)

        # m = torch.distributions.Normal(weights[:, 0].float(), weights[:, 1].float())
        m = torch.distributions.Beta(weights[:, 0].float(), weights[:, 1].float())
        log_probs = m.log_prob(action_batch)
        # beta_distribution = torch.distributions.Normal(old_weights[:, 0].float(), old_weights[:, 1].float())
        beta_distribution = torch.distributions.Beta(old_weights[:, 0].float(), old_weights[:, 1].float())
        old_log_probs = beta_distribution.log_prob(action_batch)
        ratio = torch.exp(log_probs - old_log_probs)

        clip_ratio = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
        surrogate1 = ratio * advantage
        surrogate2 = clip_ratio * advantage
        ppo_loss = -torch.min(surrogate1, surrogate2).mean()

        value_list = []
        for state in batch.state:
            if isinstance(state, torch.Tensor):
                state = state.half().to(device)
            elif isinstance(state, list):
                state = [s.float().to(device) for s in state]
            critic_value = critic(*state).unsqueeze(0)
            value_list.append(critic_value)
        values = torch.cat(value_list, dim=0)

        returns = advantage + value
        critic_loss = (returns - values) ** 2
        critic_loss = critic_loss.mean()
        total_loss = ppo_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        critic.optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        critic.optimizer.step()



class actor(nn.Module):
    def __init__(self, policyNet, tau):
        super(actor, self).__init__()
        self.target_policy = policyNet
        self.active_policy = policyNet

    def get_target_logOutput(self, x1, x2, x3):
        out = self.target_policy(x1, x2, x3)
        logOut = torch.log(out)
        return logOut

    def get_target_output(self, x1, x2, x3, scope):
        if scope == "target":
            out = self.target_policy(x1, x2, x3)
        if scope == "active":
            out = self.active_policy(x1, x2, x3)
        return out

    def get_gradient(self, x1, x2, x3, reward, scope):
        if scope == "target":
            out = self.target_policy(x1, x2, x3)
            logout = torch.log(out).view(-1)
            index = reward.index(0)
            index = (index + 1) % 2
            # print(out, reward, index, logout[index].view(-1), logout)
            # print(logout[index].view(-1))
            grad = torch.autograd.grad(logout[index].view(-1),
                                       self.target_policy.parameters())  # torch.cuda.FloatTensor(reward[index])
            # print(grad[0].size(), grad[1].size(), grad[2].size())
            # print(grad[0], grad[1], grad[2])
            grad[0].data = grad[0].data * reward[index]
            grad[1].data = grad[1].data * reward[index]
            grad[2].data = grad[2].data * reward[index]
            # print(grad[0], grad[1], grad[2])
            return grad
        if scope == "active":
            out = self.active_policy(x1, x2, x3)
        return out

    def assign_active_network_gradients(self, grad1, grad2, grad3):
        params = [grad1, grad2, grad3]
        i = 0
        for name, x in self.active_policy.named_parameters():
            x.grad = deepcopy(params[i])
            i += 1

    def update_target_network(self):
        params = []
        for name, x in self.active_policy.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.target_policy.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1 - tau))
            i += 1

    def assign_active_network(self):
        params = []
        for name, x in self.target_policy.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.active_policy.named_parameters():
            x.data = deepcopy(params[i].data)
            i += 1


import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_dim, num_teacher, embedding_length, hidden_dim=[256, 128], output_dim=1):
        super(Critic, self).__init__()

        self.W1 = nn.Parameter(torch.FloatTensor(embedding_length, 128).uniform_(-0.5, 0.5))  # 768,128
        self.W2 = nn.Parameter(torch.FloatTensor(128, 128).uniform_(-0.5, 0.5))  # 128,128
        self.W3 = nn.Parameter(torch.FloatTensor(num_teacher, 128).uniform_(-0.5, 0.5))  # 2,128
        self.b = nn.Parameter(torch.FloatTensor(1, 128).uniform_(-0.5, 0.5))

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)

    def forward(self, x1, x2, x3):
        # x1: (num_sent, batch_size, embedding_length) 2,128,768
        # x2: (num_teacher, batch_size, batch_size) 2,128,128
        # x3: (num_teacher, 1) 1,2

        x1_ = torch.matmul(x1, self.W1)  # 2,128,128
        x2_ = torch.matmul(x2, self.W2)  # 2,128,128

        x3_ = torch.matmul(x3, self.W3)  # 1,128

        scaled_out = torch.sigmoid(x1_ + x2_ + x3_ + self.b)

        scaled_out = torch.clamp(scaled_out, min=1e-5, max=1 - 1e-5)  # 2,128,128

        scaled_out_reshaped = scaled_out.view(-1, 128)  # 形状现在是 [-1, 128]

        critic_out = self.model(scaled_out_reshaped)
        critic_out = critic_out.mean()
        return critic_out
