import torch.nn as nn
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, archi_shape, archi_actionvation_fn, archi_input_size, archi_output_size, dist_init_std, device='cpu'):
        super(Actor, self).__init__()

        self.activation_fn = archi_actionvation_fn

        self.fc1 = nn.Linear(archi_input_size, archi_shape[0]).to(device)
        self.fc2 = nn.Linear(archi_shape[0], archi_shape[1]).to(device)
        self.fc3 = nn.Linear(archi_shape[1], archi_output_size).to(device)

        scale = [np.sqrt(2), np.sqrt(2), np.sqrt(2)]
        torch.nn.init.orthogonal_(self.fc1.weight, gain=scale[0])
        torch.nn.init.orthogonal_(self.fc1.weight, gain=scale[1])
        torch.nn.init.orthogonal_(self.fc1.weight, gain=scale[2])

        self.input_shape = [archi_input_size]
        self.output_shape = [archi_output_size]

        self.dist_dim = archi_output_size
        self.dist_std = nn.Parameter(dist_init_std * torch.ones(self.dist_dim, device=device))

        self.device = device

    def forward(self, obs):
        x = self.fc1(obs)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)

        return self.fc3(x)

    def sample(self, obs):
        # with torch.no_grad(): <--- it does not affect FPS
        action_mean = self.forward(obs)
        distribution = Normal(action_mean, self.dist_std.reshape(self.dist_dim))
        action_sample = distribution.sample()
        log_prob = distribution.log_prob(action_sample).sum(dim=1)

        return action_sample.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        action_mean = self.forward(obs)
        distribution = Normal(action_mean, self.dist_std.reshape(self.dist_dim))
        actions_log_prob = distribution.log_prob(actions).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def enforce_minimum_std(self, min_std):
        current_std = self.dist_std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.dist_std.data = new_std

    def deterministic_parameters(self):
        return self.parameters()

    @property
    def obs_shape(self):
        return self.input_shape

    @property
    def action_shape(self):
        return self.output_shape


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

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
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
