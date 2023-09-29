import numpy as np
import random
import torch
from External import *
import os
from DataLoader import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
flag_cuda = torch.cuda.is_available()
from Post_feat_fusion import Concate_fusion

SEED = 9159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministics = True
torch.backends.cudnn.benchmark = False

data_type = 'DWI'
doc = ['ad', 'fa', 'md', 'rd']
Extra = ['Sex_Scanner']
Data_Aug = ['Normalize']
weight_decay = [0]
Loss = ['MSE']
learning_rate = 1e-3
iter = 801
wd = 0.001
Num_worker = 10
n_splits = 5

add = './Results/post_fusion_NonMO/'
if not os.path.exists(add):
    os.makedirs(add)
print('fusion--stage')
for temp_id in range(n_splits):
    ## run each split
    split = temp_id
    path = './Results/' + data_type + '_Data_split_{}.pt'.format(split)
    Y_train, Y_val, Y_test, Y_Followbase, Y_Unhealthy, Y_Followup = Y_supervision_load(path)

    for id1 in range(len(Extra)):
        for id2 in range(len(Data_Aug)):
            for id3 in range(len(weight_decay)):
                for id4 in range(len(Loss)):
                    ## load embed feature
                    train = []
                    val = []
                    test = []
                    Followbase = []
                    Unhealthy = []
                    Followup = []
                    for id5 in range(len(doc)):
                        temp_train, temp_val, temp_test, temp_Followbase, temp_Unhealthy, temp_Followup = load_embed_feat(split, doc[id5], Extra[id1], Data_Aug[id2], weight_decay[id3], Loss[id4])
                        train.append(temp_train)
                        val.append(temp_val)
                        test.append(temp_test)
                        Followbase.append(temp_Followbase)
                        Unhealthy.append(temp_Unhealthy)
                        Followup.append(temp_Followup)

                    print('Data_type:', data_type, 'Data Augmentation ', Data_Aug[id2], ' Use ', Extra[id1], ' Loss:', Loss[id4])
                    trainloader = get_data_loader(post_concat(train, Y_train), batch_size=256, num_workers=Num_worker)
                    valloader = get_data_loader(post_concat(val, Y_val), batch_size=256, num_workers=Num_worker)
                    testloader = get_data_loader(post_concat(test, Y_test), batch_size=256, num_workers=Num_worker)
                    followbaseloader = get_data_loader(post_concat(Followbase, Y_Followbase), batch_size=256, num_workers=Num_worker)
                    unhealthyloader = get_data_loader(post_concat(Unhealthy, Y_Unhealthy), batch_size=256, num_workers=Num_worker)
                    followuploader = get_data_loader(post_concat(Followup, Y_Followup), batch_size=256, num_workers=Num_worker)
                    Concate_fusion(flag_cuda, learning_rate, wd, trainloader, valloader, testloader, followbaseloader, unhealthyloader, followuploader,
                                        data_type, Extra[id1], Data_Aug[id2], Loss[id4], split, add).process(iter)

                    ## validation and test
                    print('Validation_test')
                    Concate_fusion(flag_cuda, learning_rate, wd, trainloader, valloader, testloader, followbaseloader,
                                unhealthyloader, followuploader, data_type, Extra[id1], Data_Aug[id2], Loss[id4], split, add).Validation_test(iter)


                    ## other extra data, e.g., followup, unhealthy, badimage
                    print('follow base, unhealthy, follow up')
                    Concate_fusion(flag_cuda, learning_rate, wd, trainloader, valloader, testloader, followbaseloader,
                                unhealthyloader, followuploader, data_type, Extra[id1], Data_Aug[id2], Loss[id4], split, add).Extra_process(iter)