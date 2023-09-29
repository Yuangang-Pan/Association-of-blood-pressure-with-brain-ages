import numpy as np
import random
import torch
from External import *
import os
from DataLoader import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
flag_cuda = torch.cuda.is_available()
from main import Robust_regression

SEED = 9159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministics = True
torch.backends.cudnn.benchmark = False

isGadi = True
root_DWI = '/g/data/ey6/UKBB_DWI_20210409/'
root_DWI_Full = ''
root_T1 = ''
if not os.path.exists(root_DWI):
    isGadi = False
    root_DWI = '/home/yuangang/Yuangang-BrainAge/'

# DWI_doc = ['ad', 'fa', 'md', 'mo', 'rd']
DWI_doc = ['fa']


DWI_doc_Full = ['AD_warped.nii.gz', 'FA_warped.nii.gz', 'MD_warped.nii.gz', 'RD_warped.nii.gz']
T1_doc = ['gm_mni', 'T1_brain_to_MNI', 'wm_mni']

type = 'DWI'
if type == 'DWI':
    root = root_DWI
    doc = DWI_doc
elif type == 'DWI_Full':
    root = root_DWI_Full
    doc = DWI_doc_Full
else:
    root = root_T1
    doc = T1_doc

# Extra = ['No_Sex_Scanner', 'Sex_Scanner']
Extra = ['Sex_Scanner']
# Data_Aug = ['None', 'Normalize', 'RandomHorizontalFlip']
Data_Aug = ['Normalize']
# Loss = ['MSE', 'SmoothL1']
Loss = ['MSE']
split = [0.6, 0.2, 0.2]
batch_size = 32
learning_rate = 1e-3
weight_decay = [0]
iter = 301
Num_worker = 10
path = './Results/Data_split.pt'
# Y, Y_Followbase, Y_Unhealthy, Y_Followup = load_Supervision()
# Y_train, Y_val, Y_test = training_data_preparation(Y, isGadi, split)
# # state = {'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test, 'Followbase': Y_Followbase, 'Unhealthy': Y_Unhealthy, 'Followup': Y_Followup}
# # torch.save(state, path)
Y_train, Y_val, Y_test, Y_Followbase, Y_Unhealthy, Y_Followup = Y_supervision_load(path)

print('train_val_test:', split, ' batch_size:', batch_size, ' learning_rate:', learning_rate, ' iter:', iter)
print('Data Size \ttrain:{} \tval:{} \ttest:{}'.format(Y_train.shape[0], Y_val.shape[0], Y_test.shape[0]))
test_state = True
extra_state = True
for idx in range(len(doc)):
    for id1 in range(len(Data_Aug)):
        for id2 in range((len(Extra))):
            for id3 in range(len(weight_decay)):
                for id4 in range(len(Loss)):
                    ## training
                    valloader = []
                    testloader = []
                    followbaseloader = []
                    unhealthyloader = []
                    followuploader = []
                    trainset, X_mean, X_std = get_train_dataset(Y_train, type, root, doc[idx], Data_Aug[id1])
                    add = './Results/' + doc[idx] + '/'
                    torch.save(X_mean, add + 'X_mean.pt')
                    torch.save(X_std, add + 'X_std.pt')
                    trainloader = get_data_loader(trainset, batch_size, num_workers=Num_worker)
                    print('Data_type:', type, ' Data:', doc[idx], ' Data Augmentation ', Data_Aug[id1], ' Use ',
                          Extra[id2], ' Weight Decay:', weight_decay[id3], ' Loss:', Loss[id4])
                    Robust_regression(flag_cuda, learning_rate, weight_decay[id3], trainloader, valloader, testloader,
                                      followbaseloader, unhealthyloader, followuploader,
                                      type, Extra[id2], Data_Aug[id1], doc[idx], Loss[id4]).process(iter)

                    del trainset, trainloader
                    trainloader = []
                    ## validation and test
                    if test_state:
                        print('Validation_test')
                        X_mean = torch.load(add + 'X_mean.pt')
                        X_std = torch.load(add + 'X_std.pt')
                        valset, testset = get_val_dataset(Y_val, Y_test, type, root, doc[idx], Data_Aug[id1], X_mean, X_std)
                        valloader = get_data_loader(valset, batch_size, num_workers=Num_worker)
                        testloader = get_data_loader(testset, batch_size, num_workers=Num_worker)
                        Robust_regression(flag_cuda, learning_rate, weight_decay[id3], trainloader, valloader, testloader, followbaseloader, unhealthyloader, followuploader,
                                          type, Extra[id2], Data_Aug[id1], doc[idx], Loss[id4]).Validation_test(iter)

                        del valset, testset, valloader, testloader
                        valloader = []
                        testloader = []
                        ## other extra data, e.g., followup, unhealthy, badimage
                        if extra_state:
                            print('follow base, unhealthy, follow up')
                            followbaseset, unhealthset, followupset = get_extra_dataset(Y_Followbase, Y_Unhealthy, Y_Followup, type, root, doc[idx], Data_Aug[id1], X_mean, X_std)
                            followbaseloader = get_data_loader(followbaseset, batch_size, num_workers=Num_worker)
                            unhealthyloader = get_data_loader(unhealthset, batch_size, num_workers=Num_worker)
                            followuploader = get_data_loader(followupset, batch_size, num_workers=Num_worker)
                            Robust_regression(flag_cuda, learning_rate, weight_decay[id3], trainloader, valloader, testloader, followbaseloader, unhealthyloader, followuploader,
                                              type, Extra[id2], Data_Aug[id1], doc[idx], Loss[id4]).Extra_process(iter)
                            del followbaseset, unhealthset, followupset, followbaseloader, unhealthyloader, followuploader
                            followbaseloader = []
                            unhealthyloader = []
                            followuploader = []