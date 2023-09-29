from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import torch
from DataLoader import *

import torchvision.transforms as transforms
train_transfer = transforms.Compose([
    transforms.RandomHorizontalFlip(0.25)
    # transforms.RandomVerticalFlip(0.5)
    ])
test_transfer = transforms.Compose([])

class MyDataset():
    def __init__(self, Data, Label, data_transfer):
        self.transforms = data_transfer
        self.Data = Data
        self.Label = Label

    def __getitem__(self, Id):
        x, y = self.Data[Id,:,:,:], self.Label[Id,:]

        if self.transforms:
            x = self.transforms(x)
            # x = torch.from_numpy(x)
        return x, y

    def __len__(self):
        return self.Label.shape[0]


def get_train_dataset(Y_train, type, root, doc, Augment):
    X_mean = 0
    X_std = 0
    Y_train = torch.from_numpy(Y_train.to_numpy())

    if type == 'DWI' or type == 'DWI_Full':
        ID_train = Y_train[:, 0].int().data.numpy()
        idx_train, X_train = load_DWI_NeuroImage(ID_train, root, doc)
    else:
        ID_train = Y_train[:, 0].int().data.numpy()
        idx_train, X_train = load_T1_NeuroImage(ID_train, root, doc)

    X_train = torch.from_numpy(X_train)
    Y_train = Y_train[idx_train,:]
    if Augment == 'None':
        traindata = data_utils.TensorDataset(X_train, Y_train)
    elif Augment == 'Normalize':
        X_mean, X_std = X_train.mean(), X_train.std()
        X_train = (X_train - X_mean) / X_std
        traindata = data_utils.TensorDataset(X_train, Y_train)
    else:
        traindata = MyDataset(X_train, Y_train, train_transfer)

    return traindata, X_mean, X_std

def get_val_dataset(Y_val, Y_test, type, root, doc, Augment, X_mean, X_std):

    Y_val = torch.from_numpy(Y_val.to_numpy())
    Y_test = torch.from_numpy(Y_test.to_numpy())

    if type == 'DWI' or type == 'DWI_Full':
        ID_val = Y_val[:, 0].int().data.numpy()
        ID_test = Y_test[:, 0].int().data.numpy()
        idx_val, X_val = load_DWI_NeuroImage(ID_val, root, doc)
        idx_test, X_test = load_DWI_NeuroImage(ID_test, root, doc)
    else:
        ID_val = Y_val[:, 0].int().data.numpy()
        ID_test = Y_test[:, 0].int().data.numpy()
        idx_val, X_val = load_T1_NeuroImage(ID_val, root, doc)
        idx_test, X_test = load_T1_NeuroImage(ID_test, root, doc)

    X_val = torch.from_numpy(X_val)
    Y_val = Y_val[idx_val, :]
    X_test = torch.from_numpy(X_test)
    Y_test = Y_test[idx_test, :]
    if Augment == 'None':
        valdata = data_utils.TensorDataset(X_val, Y_val)
        testdata = data_utils.TensorDataset(X_test, Y_test)
    elif Augment == 'Normalize':
        X_val = (X_val - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        valdata = data_utils.TensorDataset(X_val, Y_val)
        testdata = data_utils.TensorDataset(X_test, Y_test)
    else:
        X_val = (X_val - X_mean) / X_std
        valdata = MyDataset(X_val, Y_val, test_transfer)

        X_test = (X_test - X_mean) / X_std
        testdata = MyDataset(X_test, Y_test, test_transfer)
    return valdata, testdata

def get_extra_dataset(Y_Followbase, Y_Unhealthy, Y_Followup, type, root, doc, Augment, X_mean, X_std):

    Y_base = torch.from_numpy(Y_Followbase.to_numpy())
    Y_unh = torch.from_numpy(Y_Unhealthy.to_numpy())
    Y_up = torch.from_numpy(Y_Followup.to_numpy())

    followup = True
    if type == 'DWI' or type == 'DWI_Full':
        ID_base = Y_base[:, 0].int().data.numpy()
        ID_unh = Y_unh[:, 0].int().data.numpy()
        ID_up = Y_up[:, 0].int().data.numpy()
        idx_base, X_base = load_DWI_NeuroImage(ID_base, root, doc)
        idx_unh, X_unh = load_DWI_NeuroImage(ID_unh, root, doc)
        idx_up, X_up = load_DWI_NeuroImage(ID_up, root, doc, followup)
    else:
        ID_base = Y_base[:, 0].int().data.numpy()
        ID_unh = Y_unh[:, 0].int().data.numpy()
        ID_up = Y_up[:, 0].int().data.numpy()
        idx_base, X_base = load_T1_NeuroImage(ID_base, root, doc)
        idx_unh, X_unh = load_T1_NeuroImage(ID_unh, root, doc)
        idx_up, X_up = load_T1_NeuroImage(ID_up, root, doc)

    X_base = torch.from_numpy(X_base)
    Y_base = Y_base[idx_base, :]
    X_unh = torch.from_numpy(X_unh)
    Y_unh = Y_unh[idx_unh, :]
    X_up = torch.from_numpy(X_up)
    Y_up = Y_up[idx_up, :]
    if Augment == 'None':
        basedata = data_utils.TensorDataset(X_base, Y_base)
        unhdata = data_utils.TensorDataset(X_unh, Y_unh)
        updata = data_utils.TensorDataset(X_up, Y_up)
    elif Augment == 'Normalize':
        X_base = (X_base - X_mean) / X_std
        X_unh = (X_unh - X_mean) / X_std
        X_up = (X_up - X_mean) / X_std
        basedata = data_utils.TensorDataset(X_base, Y_base)
        unhdata = data_utils.TensorDataset(X_unh, Y_unh)
        updata = data_utils.TensorDataset(X_up, Y_up)
    else:
        X_base = (X_base - X_mean) / X_std
        basedata = MyDataset(X_base, Y_base, test_transfer)

        X_unh = (X_unh - X_mean) / X_std
        unhdata = MyDataset(X_unh, Y_unh, test_transfer)

        X_up = (X_up - X_mean) / X_std
        updata = MyDataset(X_up, Y_up, test_transfer)
    return basedata, unhdata, updata

def get_data_loader(dataset, batch_size=10, num_workers=20):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def post_concat(dataset, Y_sup):
    Y_sup = torch.from_numpy(Y_sup.to_numpy())
    idx = np.argsort(Y_sup[:, 0])
    Y_sup = Y_sup[idx, :]

    feat = []
    for id in range(len(dataset)):
        feat.append(dataset[id][:,2:])
    feat = torch.from_numpy(np.concatenate(feat, 1))
    data = data_utils.TensorDataset(feat.float(), Y_sup.float())
    return data
