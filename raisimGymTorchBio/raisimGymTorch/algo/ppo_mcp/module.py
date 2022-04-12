import torch.nn as nn
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, dim_action, dim_obs_stat, dim_obs_goal, num_prmtv, device='cpu'):
        super(Actor, self).__init__()

        self.dim_action = dim_action
        self.dim_obs_stat = dim_obs_stat
        self.dim_obs_goal = dim_obs_goal
        self.num_prmtv = num_prmtv
        self.device = device

        # parameters for primitive network
        self.prmtv_fc1 = nn.Linear(self.dim_obs_stat, 512).to(device)
        self.prmtv_fc2 = nn.Linear(512, 256).to(device)
        self.prmtv_fci_x1 = list()
        self.prmtv_fci_x2 = list()
        for idx in range(self.num_prmtv):
            self.prmtv_fci_x1.append(nn.Linear(256, 256).to(device))
            self.prmtv_fci_x2.append(nn.Linear(256, self.num_prmtv*self.dim_action*2).to(device))

        # parameters for gating network
        self.gtfcn_stat_fc1 = nn.Linear(self.dim_obs_stat, 512).to(device)
        self.gtfcn_stat_fc2 = nn.Linear(512, 256).to(device)
        self.gtfcn_stat_fc3 = nn.Linear(256, 128).to(device)
        self.gtfcn_goal_fc1 = nn.Linear(self.dim_obs_goal, 512).to(device)
        self.gtfcn_goal_fc2 = nn.Linear(512, 256).to(device)
        self.gtfcn_goal_fc3 = nn.Linear(256, 128).to(device)
        self.gtfcn_fc3      = nn.Linear(256, self.num_prmtv).to(device)

    def net_prmtv(self, obs_stat):
        s1 = self.prmtv_fc1(obs_stat)
        s1 = F.leaky_relu(s1)
        s1 = self.prmtv_fc2(s1)
        s1 = F.leaky_relu(s1)

        ps_list = list()
        for idx in range(self.num_prmtv):
            x = self.prmtv_fci_x1[idx](s1)
            x = F.leaky_relu(x)
            x = self.prmtv_fci_x2[idx](x)
            ps_list.append(x)

        ps = torch.stack(ps_list, dim=2)
        mu_prmtv = ps[:, :20, :]
        sigma_prmtv_temp = ps[:, 20:, :]
        sigma_prmtv = torch.exp(sigma_prmtv_temp)  # to make them all positive values

        return mu_prmtv, sigma_prmtv

    def net_gate(self, obs_stat, obs_goal):
        s2 = self.gtfcn_stat_fc1(obs_stat)
        s2 = F.leaky_relu(s2)
        s2 = self.gtfcn_stat_fc2(s2)
        s2 = F.leaky_relu(s2)
        s2 = self.gtfcn_stat_fc3(s2)

        g2 = self.gtfcn_goal_fc1(obs_goal)
        g2 = F.leaky_relu(g2)
        g2 = self.gtfcn_goal_fc2(g2)
        g2 = F.leaky_relu(g2)
        g2 = self.gtfcn_goal_fc3(g2)

        sg = torch.cat((s2, g2), 1)
        sg = F.leaky_relu(sg)
        w_gate_temp = self.gtfcn_fc3(sg)
        w_gate = torch.exp(w_gate_temp)  # to make them all positive values

        return w_gate

    def sample(self, obs):
        obs_stat = obs[:, :self.dim_obs_stat]
        obs_goal = obs[:, self.dim_obs_stat:]

        # with torch.no_grad():
        mu_prmtv, sigma_prmtv = self.net_prmtv(obs_stat)
        w_gate = self.net_gate(obs_stat, obs_goal)

        w_gate_repeat = w_gate.unsqueeze(1).repeat(1, self.dim_action, 1)
        sum1 = torch.sum(mu_prmtv * w_gate_repeat / sigma_prmtv, dim=2)
        sum2 = torch.sum(w_gate_repeat / sigma_prmtv, dim=2)
        mu_net = sum1 / sum2
        sigma_net = 1 / sum2

        distribution = Normal(mu_net, sigma_net)
        action_sample = distribution.sample()  # actions
        log_prob = distribution.log_prob(action_sample).sum(dim=1)

        return action_sample.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        obs_stat = obs[:, :self.dim_obs_stat]
        obs_goal = obs[:, self.dim_obs_stat:]

        mu_prmtv, sigma_prmtv = self.net_prmtv(obs_stat)   # requires_grad
        w_gate = self.net_gate(obs_stat, obs_goal)         # requires_grad

        w_gate_repeat = w_gate.unsqueeze(1).repeat(1, self.dim_action, 1)
        sum1 = torch.sum(mu_prmtv * w_gate_repeat / sigma_prmtv, dim=2)
        sum2 = torch.sum(w_gate_repeat / sigma_prmtv, dim=2)
        mu_net = sum1 / sum2
        sigma_net = 1 / sum2

        distribution = Normal(mu_net, sigma_net)
        actions_log_prob = distribution.log_prob(actions).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    # def noiseless_action(self, obs):
    #     return self.architecture.architecture(torch.from_numpy(obs).to(self.device))

    def deterministic_parameters(self):
        return self.parameters()

    @property
    def obs_shape(self):
        return [self.dim_obs_stat + self.dim_obs_goal]

    @property
    def action_shape(self):
        return [self.dim_action]


class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape) - 1):
            modules.append(nn.Linear(shape[idx], shape[idx + 1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None

    def sample(self, logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
