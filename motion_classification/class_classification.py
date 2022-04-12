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


class classNet(nn.Module):
    def __init__(self, len_input):
        super(classNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(len_input)), 128),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, input_data):
        decision = self.model(input_data)
        return torch.squeeze(decision)


class classify_net:
    def __init__(self, ndim, device='cpu', learning_rate=5e-4):
        self.ndim = ndim
        self.expert_data_1 = np.array([])
        self.expert_data_2 = np.array([])
        self.expert_data_3 = np.array([])

        self.classnet = classNet(ndim)
        self.classnet.to(device)
        self.optim_learning_rate = learning_rate
        self.device = device
        self.nsample = 1000   # use only this number of samples from each agent and expert for a update

        # self.optimizer = torch.optim.AdamW(self.class.parameters(), lr=self.optim_learning_rate, betas=self.optim_betas)
        self.optimizer = torch.optim.SGD(self.classnet.parameters(), lr=self.optim_learning_rate, momentum=0.9)
        self.classify_loss = torch.nn.CrossEntropyLoss()
        self.classify_loss.to(device)

    def update(self):

        # obtain random samples from expert_data
        idxrandom_expert_1 = np.random.permutation(self.expert_data_1.shape[0])
        idxrandom_expert_2 = np.random.permutation(self.expert_data_2.shape[0])
        idxrandom_expert_3 = np.random.permutation(self.expert_data_3.shape[0])
        expert_data_samples_1 = self.expert_data_1[idxrandom_expert_1[:self.nsample], :]   # random selection
        expert_data_samples_2 = self.expert_data_2[idxrandom_expert_2[:self.nsample], :]
        expert_data_samples_3 = self.expert_data_3[idxrandom_expert_3[:self.nsample], :]

        expert_data_pytorch_tensor_1 = torch.from_numpy(expert_data_samples_1).float().to(self.device)
        expert_data_pytorch_tensor_2 = torch.from_numpy(expert_data_samples_2).float().to(self.device)
        expert_data_pytorch_tensor_3 = torch.from_numpy(expert_data_samples_3).float().to(self.device)
        expert_data_pytorch_tensor_1.requires_grad = False
        expert_data_pytorch_tensor_2.requires_grad = False
        expert_data_pytorch_tensor_3.requires_grad = False

        label_1 = torch.Tensor([1, 0, 0])
        label_1 = label_1.repeat(self.nsample, 1)
        label_2 = torch.Tensor([0, 1, 0])
        label_2 = label_2.repeat(self.nsample, 1)
        label_3 = torch.Tensor([0, 0, 1])
        label_3 = label_3.repeat(self.nsample, 1)

        # self.optimizer.zero_grad()
        # expert_loss_1 = self.classify_loss(self.classnet(expert_data_pytorch_tensor_1), label_1.to(self.device))
        # expert_loss_2 = self.classify_loss(self.classnet(expert_data_pytorch_tensor_2), label_2.to(self.device))
        # expert_loss_3 = self.classify_loss(self.classnet(expert_data_pytorch_tensor_3), label_3.to(self.device))
        #
        # expert_loss = expert_loss_1 + expert_loss_2 + expert_loss_3
        #
        # expert_loss.backward()
        # self.optimizer.step()

        self.optimizer.zero_grad()
        expert_loss_1 = self.classify_loss(self.classnet(expert_data_pytorch_tensor_1), label_1.to(self.device))
        expert_loss_1.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        expert_loss_2 = self.classify_loss(self.classnet(expert_data_pytorch_tensor_2), label_2.to(self.device))
        expert_loss_2.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        expert_loss_3 = self.classify_loss(self.classnet(expert_data_pytorch_tensor_3), label_3.to(self.device))
        expert_loss_3.backward()
        self.optimizer.step()

        # return expert_loss.item()
        return expert_loss_1.item(), expert_loss_2.item(), expert_loss_3.item()

    def test(self, input1, input2):

        input_amp = np.concatenate((input1, input2), axis=1)
        input_amp = torch.from_numpy(input_amp).float().to(self.device)

        output = self.classnet(input_amp)
        # print(output)

        _, predicted = torch.max(output, 0)


        return predicted.item()


    def load_refmotion_data_1(self, filepath_refmotion_json):
        with open(filepath_refmotion_json) as f:
            json_data = json.load(f)
        expert_data = np.array(json_data['expert'])
        # expert_data = np.array(eval(json_data['agent']))
        assert expert_data.shape[1] == self.ndim, "Dimension mismatch in expert record"
        assert expert_data.shape[0] > 1000, "Size too small in expert record"
        self.expert_data_1 = expert_data

    def load_refmotion_data_2(self, filepath_refmotion_json):
        with open(filepath_refmotion_json) as f:
            json_data = json.load(f)
        expert_data = np.array(json_data['expert'])
        # expert_data = np.array(eval(json_data['agent']))
        assert expert_data.shape[1] == self.ndim, "Dimension mismatch in expert record"
        assert expert_data.shape[0] > 1000, "Size too small in expert record"
        self.expert_data_2 = expert_data

    def load_refmotion_data_3(self, filepath_refmotion_json):
        with open(filepath_refmotion_json) as f:
            json_data = json.load(f)
        expert_data = np.array(json_data['expert'])
        # expert_data = np.array(eval(json_data['agent']))
        assert expert_data.shape[1] == self.ndim, "Dimension mismatch in expert record"
        assert expert_data.shape[0] > 1000, "Size too small in expert record"
        self.expert_data_3 = expert_data


    def optimizer_reset(self):
        self.optimizer = torch.optim.SGD(self.classnet.parameters(), lr=self.optim_learning_rate, momentum=0.9)


    def save_state(self, filepath):
        torch.save({
            'architecture_state_dict': self.classnet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            filepath)


    def load_state(self, filepath):
        checkpoint = torch.load(filepath)
        self.classnet.load_state_dict(checkpoint['architecture_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cla_1 = classify_net(142, device=device, learning_rate=0.001)
    cla_1.load_refmotion_data_1("/Works/mingi/raisim_v6_workspace/raisimLib/raisimGymMeta7/raisimGymTorch/env/envs/rsg_gaitmsk_MAML/rsc/expert/AMP_subject01_aug_v2.json")
    idxrandom_expert_1 = np.random.permutation(cla_1.expert_data_1.shape[0])
    expert_data_samples_1 = cla_1.expert_data_1[idxrandom_expert_1[:cla_1.nsample], :]   # random selection

    expert_data_pytorch_tensor_1 = torch.from_numpy(expert_data_samples_1).float().to(cla_1.device)
    expert_data_pytorch_tensor_1.requires_grad = False

    label_1 = torch.Tensor([1, 0, 0])
    label_1 = label_1.repeat(cla_1.nsample, 1)

    cla_1.optimizer.zero_grad()
    print(cla_1.classnet(expert_data_pytorch_tensor_1))
    print(label_1.to(cla_1.device))

    expert_loss_1 = cla_1.classify_loss(cla_1.classnet(expert_data_pytorch_tensor_1), label_1.to(cla_1.device))
    print(expert_loss_1)
