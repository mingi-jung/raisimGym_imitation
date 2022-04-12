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

classifier.load_refmotion_data_1("/Works/mingi/raisim_v6_workspace/raisimLib/raisimGymMeta7/raisimGymTorch/env/envs/rsg_gaitmsk_MAML/rsc/expert/AMP_subject04_aug_v2.json")
classifier.load_refmotion_data_2("/Works/mingi/raisim_v6_workspace/raisimLib/raisimGymMeta7/raisimGymTorch/env/envs/rsg_gaitmsk_MAML/rsc/expert/AMP_subject06_aug_v2.json")
classifier.load_refmotion_data_3("/Works/mingi/raisim_v6_workspace/raisimLib/raisimGymMeta7/raisimGymTorch/env/envs/rsg_gaitmsk_MAML/rsc/expert/AMP_subject10_aug_v2.json")


classifier_loss = 0.0

for update in range(10000):

    if update % 500 == 0:
        classifier.save_state(os.path.abspath(task_path + "/data/full_classifier_" + str(update) + '.pt'))

    # classifier_loss = classifier.update()
    classifier_loss_1, classifier_loss_2, classifier_loss_3 = classifier.update()

    if update % 50 == 0:
        print('{:>6}th iteration'.format(update))
        # classifier_loss = (classifier_loss_1 + classifier_loss_2 + classifier_loss_3) / 3
        # print('{:<40} {:>6}'.format("classifier_loss: ", '{:6.4f}'.format(classifier_loss)))
        print('{:<40} {:>6}'.format("classifier_loss: ", '{:6.4f}'.format(classifier_loss_1)))
        print('{:<40} {:>6}'.format("classifier_loss: ", '{:6.4f}'.format(classifier_loss_2)))
        print('{:<40} {:>6}'.format("classifier_loss: ", '{:6.4f}'.format(classifier_loss_3)))
