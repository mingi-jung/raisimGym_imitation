# 2021.06.04 SKOO MSKBIODYN@KAIST###

# import os
import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import class_classification

import os



task_path = os.path.dirname(os.path.realpath(__file__))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier = class_classification.classify_net(142, device=device, learning_rate=0.001)
classifier.load_state("/Works/mingi/raisim_v6_workspace/raisimLib/raisimGymMeta7/classification/data/full_classifier_9500.pt")
classifier.load_refmotion_data_1("/Works/mingi/raisim_v6_workspace/raisimLib/raisimGymMeta7/raisimGymTorch/env/envs/rsg_gaitmsk_MAML/rsc/expert/AMP_subject06_aug_v2.json")

classifier_loss = 0.0

total = 0
correct = 0

for update in range(10000):

    idxrandom_expert_1 = np.random.permutation(classifier.expert_data_1.shape[0])
    expert_data_samples_1 = classifier.expert_data_1[idxrandom_expert_1[:1], :]
    expert_data_pytorch_tensor_1 = torch.from_numpy(expert_data_samples_1).float().to(classifier.device)
    expert_data_pytorch_tensor_1.requires_grad = False

    output = classifier.classnet(expert_data_pytorch_tensor_1)
    # print(output)
    _, predicted = torch.max(output, 0)
    # print(predicted.item())

    total += 1
    if predicted.item() == 0:
        correct += 1

    print(f'Accuracy: {100 * correct // total} %')
