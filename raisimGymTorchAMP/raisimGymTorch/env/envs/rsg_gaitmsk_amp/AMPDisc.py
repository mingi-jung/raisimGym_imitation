# 2021.06.04 SKOO MSKBIODYN@KAIST###

# import os
import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
# from .storage import RolloutStorage


class DiscNet(nn.Module):
    def __init__(self, len_input):
        super(DiscNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(len_input)), 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, input_data):
        decision = self.model(input_data)
        return torch.squeeze(decision)


class AMPDisc:
    def __init__(self, ndim, num_envs, device='cpu', learning_rate=5e-4, betas=(0.5, 0.999)):

        self.ndim = ndim
        self.nenv = num_envs
        self.agent_data = list()
        self.expert_data = np.array([])

        self.disc = DiscNet(ndim)
        self.disc.to(device)
        self.optim_learning_rate = learning_rate
        self.optim_betas = betas

        self.device = device
        self.nsample = 4096   # use only this number of samples from each agent and expert for a update

        self.optimizer = torch.optim.AdamW(self.disc.parameters(), lr=self.optim_learning_rate, betas=self.optim_betas)
        self.adversarial_loss = torch.nn.MSELoss()
        self.adversarial_loss.to(device)


    def observe(self, obs0, obs1):
        # num_envs of observation [obs0, obs1] ... like (num_envs) x (ndim*2) are ready
        # test the obs using the disc NN
        # save the observations in observation storage

        agent_data = np.concatenate((obs0, obs1), axis=1)
        self.agent_data.append(agent_data)

        agent_data_pytorch_tensor = torch.from_numpy(agent_data).float().to(self.device)

        # it should be 0 if discriminator performs better than the generator
        # if generator is better than the discriminator then it tends to be 1
        decision = self.disc(agent_data_pytorch_tensor)
        decision_ndarray = decision.detach().cpu().numpy()
        if decision_ndarray.ndim == 0 and decision_ndarray.size == 1:
            decision_ndarray.shape = (1,)

        return decision_ndarray


    def update(self):
        nstep = len(self.agent_data)  # nstep x nenv x ndim
        # number of all collected agent samples = nstep * self.nenv

        # obtain agent_data
        agent_data = np.zeros((nstep * self.nenv, self.ndim))
        for istep in range(nstep):
            agent_data[istep*self.nenv:(istep+1)*self.nenv, :] = self.agent_data[istep]

        idxrandom_agent = np.random.permutation(nstep * self.nenv)
        agent_data_samples = agent_data[idxrandom_agent[:self.nsample], :]

        # obtain random samples from expert_data
        idxrandom_expert = np.random.permutation(self.expert_data.shape[0])
        expert_data_samples = self.expert_data[idxrandom_expert[:self.nsample], :]   # random selection

        agent_data_pytorch_tensor = torch.from_numpy(agent_data_samples).float().to(self.device)
        agent_data_pytorch_tensor.requires_grad = False
        expert_data_pytorch_tensor = torch.from_numpy(expert_data_samples).float().to(self.device)
        expert_data_pytorch_tensor.requires_grad = False

        self.optimizer.zero_grad()
        agent_loss = self.adversarial_loss(self.disc(agent_data_pytorch_tensor), -1*torch.ones(self.nsample).to(self.device))
        expert_loss = self.adversarial_loss(self.disc(expert_data_pytorch_tensor), torch.ones(self.nsample).to(self.device))

        gradient_penalty = self.gradient_penalty(agent_data_pytorch_tensor, expert_data_pytorch_tensor)

        d_loss = agent_loss + expert_loss + 10*gradient_penalty
        d_loss.backward()
        self.optimizer.step()

        self.agent_data.clear()

        return d_loss.item(), agent_loss.item(), expert_loss.item(), gradient_penalty.item()

        # get observations from observation storage
        # select samples (num_envs) from the refmotion storage
        # train (or update) the disc NN using the observation and refmotion samples
        #
        #
        # last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        #
        # # Learning step
        # self.storage.compute_returns(last_values.to(self.device), self.gamma, self.lam)
        # mean_value_loss, mean_surrogate_loss, infos = self._train_step()
        # self.storage.clear()
        # # stop = time.time()
        #
        # if log_this_iteration:
        #     self.log({**locals(), **infos, 'ep_infos': self.ep_infos, 'ep_extrainfos': self.ep_extrainfos, 'it': update})
        #
        # self.ep_infos.clear()
        # self.ep_extrainfos.clear()


    def load_refmotion_data(self, filepath_refmotion_json):
        with open(filepath_refmotion_json) as f:
            json_data = json.load(f)
        expert_data = np.array(json_data['expert'])
        # expert_data = np.array(eval(json_data['agent']))
        assert expert_data.shape[1] == self.ndim, "Dimension mismatch in expert record"
        assert expert_data.shape[0] > 1000, "Size too small in expert record"
        self.expert_data = expert_data


    def gradient_penalty(self, agent_data_pytorch_tensor, expert_data_pytorch_tensor):
        """Calculates the gradient penalty loss"""
        # Random weight term for interpolation between real and fake samples
        alpha = np.random.random((self.nsample, 1))
        alpha_tensor = torch.from_numpy(alpha).float().to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha_tensor * agent_data_pytorch_tensor + ((1 - alpha_tensor) * expert_data_pytorch_tensor)).requires_grad_(True)
        d_interpolates = self.disc(interpolates)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            inputs=interpolates,
            outputs=d_interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty = (gradients.norm(2, dim=1) ** 2).mean()
        return gradient_penalty


    def optimizer_reset(self):
        self.optimizer = torch.optim.AdamW(self.disc.parameters(), lr=self.optim_learning_rate, betas=self.optim_betas)


    def save_state(self, filepath):
        torch.save({
            'architecture_state_dict': self.disc.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            filepath)


    def load_state(self, filepath):
        checkpoint = torch.load(filepath)
        self.disc.load_state_dict(checkpoint['architecture_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
