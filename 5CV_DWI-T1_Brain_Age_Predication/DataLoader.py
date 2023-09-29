from ast import Del
import pickle
import nibabel as nib
import os
import numpy as np
import fnmatch
import scipy.io
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

## import Subject ID
def DWI_load_Supervision():
    Path = './data_profile/UKB_DWI_Apr_9.xlsx'

    ##load the Excel and delete the bad, unhealthy or followup images
    Full_table = pd.read_excel(Path, sheet_name='baseline')
    Full_Id = Full_table['eid'].values
    ID = Full_Id
    Followup = pd.read_excel(Path, sheet_name='followup')
    Del_Followup_Id1 = Followup['eid']
    ID = np.setdiff1d(ID, Del_Followup_Id1)
    Unhealthy = pd.read_excel(Path, sheet_name='unhealthy')
    Del_Unhealthy_Id2 = Unhealthy['eid']
    ID = np.setdiff1d(ID, Del_Unhealthy_Id2)
    Bad_img = pd.read_excel(Path, sheet_name='bad_img_quality')
    Del_Badimg_Id3 = Bad_img['eid']
    ID = np.setdiff1d(ID, Del_Badimg_Id3)

    xsorted = np.argsort(Full_Id)
    ## find the index of the selected set "ID" from the full set "Full_Id"
    assert np.all(np.intersect1d(Full_Id, ID) == np.sort(ID))
    ypos = np.searchsorted(Full_Id[xsorted], ID)
    Idx = xsorted[ypos]
    ## find the index of the followup ID from the full set "Full_Id"
    assert np.all(np.intersect1d(Full_Id, Del_Followup_Id1) == np.sort(Del_Followup_Id1))
    Del_Followup_Id1 = np.intersect1d(Full_Id, Del_Followup_Id1)
    ypos = np.searchsorted(Full_Id[xsorted], Del_Followup_Id1)
    Followup_Idx = xsorted[ypos]
    ## find the index of the Unhealthy from the full set "Full_Id"
    # assert np.all(np.intersect1d(Full_Id, Del_Unhealthy_Id2) == np.sort(Del_Unhealthy_Id2))
    Del_Unhealthy_Id2 = np.intersect1d(Full_Id, Del_Unhealthy_Id2)
    ypos = np.searchsorted(Full_Id[xsorted], Del_Unhealthy_Id2)
    Unhealthy_Idx = xsorted[ypos]

    Col = ['eid', 'age', 'sex', 'scanner']
    Full_table['scanner'][Full_table['scanner'] == 11025] = 1
    Full_table['scanner'][Full_table['scanner'] == 11026] = 0
    Full_table['scanner'][Full_table['scanner'] == 11027] = -1
    Y_label = Full_table.loc[Idx, Col]
    Unhealthy_label = Full_table.loc[Unhealthy_Idx, Col]

    ## followup image contains: followbase in baseline
    Followbase_label = Full_table.loc[Followup_Idx, Col]
    ## and followup
    Followup['scanner'][Followup['scanner'] == 11025] = 1
    Followup['scanner'][Followup['scanner'] == 11026] = 0
    Followup['scanner'][Followup['scanner'] == 11027] = -1
    Followup_label = Followup.loc[:, Col]

    return Y_label, Followbase_label, Unhealthy_label, Followup_label

## import Subject ID
def T1_load_Supervision():
    Path = './data_profile/T1_Jun_10.xlsx'
    ##load the Excel and delete the unhealthy or followup images
    Full_table = pd.read_excel(Path, sheet_name='all_T1_baseline')
    Full_Id = Full_table['eid'].values
    ID = Full_Id
    Followup = pd.read_excel(Path, sheet_name='T1_followup')
    Del_Followup_Id1 = Followup['eid']
    ID = np.setdiff1d(ID, Del_Followup_Id1)
    Unhealthy = pd.read_excel(Path, sheet_name='unhealthy')
    Del_Unhealthy_Id2 = Unhealthy['eid']
    ID = np.setdiff1d(ID, Del_Unhealthy_Id2)

    xsorted = np.argsort(Full_Id)
    ## find the index of the selected set "ID" from the full set "Full_Id"
    assert np.all(np.intersect1d(Full_Id, ID) == np.sort(ID))
    ypos = np.searchsorted(Full_Id[xsorted], ID)
    Idx = xsorted[ypos]
    ## find the index of the followup ID from the full set "Full_Id"
    assert np.all(np.intersect1d(Full_Id, Del_Followup_Id1) == np.sort(Del_Followup_Id1))
    Del_Followup_Id1 = np.intersect1d(Full_Id, Del_Followup_Id1)
    ypos = np.searchsorted(Full_Id[xsorted], Del_Followup_Id1)
    Followup_Idx = xsorted[ypos]
    ## find the index of the Unhealthy from the full set "Full_Id"
    # assert np.all(np.intersect1d(Full_Id, Del_Unhealthy_Id2) == np.sort(Del_Unhealthy_Id2))
    Del_Unhealthy_Id2 = np.intersect1d(Full_Id, Del_Unhealthy_Id2)
    ypos = np.searchsorted(Full_Id[xsorted], Del_Unhealthy_Id2)
    Unhealthy_Idx = xsorted[ypos]

    baseline_Col = ['eid', 'baseline_imaging_age', 'sex', 'secanner_baseline', 'icv_baseline']
    Full_table['secanner_baseline'][Full_table['secanner_baseline'] == 11025] = 1
    Full_table['secanner_baseline'][Full_table['secanner_baseline'] == 11026] = 0
    Full_table['secanner_baseline'][Full_table['secanner_baseline'] == 11027] = -1
    ICV = Full_table['icv_baseline']
    Full_table['icv_baseline'] = (ICV - ICV.mean()) / ICV.std()
    Y_label = Full_table.loc[Idx, baseline_Col]
    Unhealthy_label = Full_table.loc[Unhealthy_Idx, baseline_Col]

    ## followup image contains: followbase in baseline
    followup_Col = ['eid', 'followup_imaging_age', 'sex', 'scanner_followup', 'icv_followup']
    Followbase_label = Full_table.loc[Followup_Idx, baseline_Col]
    ## and followup
    Followup['scanner_followup'][Followup['scanner_followup'] == 11025] = 1
    Followup['scanner_followup'][Followup['scanner_followup'] == 11026] = 0
    Followup['scanner_followup'][Followup['scanner_followup'] == 11027] = -1
    temp = Followbase_label['icv_baseline'].to_numpy()
    Followup['icv_followup'] = temp
    # ICV = Followup['icv_followup']
    # Followup['icv_followup'] = (Followup['icv_followup'] - ICV.mean()) / ICV.std()
    Followup_label = Followup.loc[:, followup_Col]

    return Y_train_label, Y_val_label, Y_test_label, Followbase_label, Unhealthy_label, Followup_label


def Y_supervision_load(path):
    state = torch.load(path)
    Y_train = state['Y_train']
    Y_val = state['Y_val']
    Y_test = state['Y_test']
    Y_Followbase = state['Followbase']
    Y_Unhealthy = state['Unhealthy']
    Y_Followup = state['Followup']
    return Y_train, Y_val, Y_test, Y_Followbase, Y_Unhealthy, Y_Followup

def training_data_preparation(Y, isGadi, ratio=[0.6, 0.2, 0.2]):
    if isGadi:
        Y_train, Rest = train_test_split(Y, test_size=1-ratio[0], shuffle=True)
        Y_val, Y_test = train_test_split(Rest, test_size=1-ratio[1]/(ratio[1]+ratio[2]), shuffle=True)
    else:
        _, Rest = train_test_split(Y, test_size=0.01, shuffle=True)
        Y_train, Rest = train_test_split(Rest, test_size=1-ratio[0], shuffle=True)
        Y_val, Y_test = train_test_split(Rest, test_size=1-ratio[1]/(ratio[1] + ratio[2]), shuffle=True)
    return Y_train, Y_val, Y_test

## load T1 train, val, test neuroimage
def load_T1_NeuroImage(Y, root, doc, followup = False):
    Data = []
    idx = []
    Num = Y.shape[0]
    for item in range(Num):
        if followup:
            data_type = '_FU_'
        else:
            data_type = '_BL_'
        file_name = 'mwp1' + Y[item].astype('str') + data_type + doc
        file_address = os.path.join(root, file_name)
        if os.path.exists(file_address):
            img = nib.load(file_address)
            temp_data = img.get_fdata().astype('float32')
            Data.append(np.reshape(temp_data, (1, temp_data.shape[0], temp_data.shape[1], temp_data.shape[2])))
            idx.append(item)
    Data = np.concatenate(Data, 0)
    return idx, Data

## load DWI train, val, test neuroimage
def load_DWI_NeuroImage(Y, root, doc, followup = False):
    Data = []
    idx = []
    Num = Y.shape[0]
    root1 = os.path.join(root, doc)
    for item in range(Num):
        end = '_native_masked_to_MNI.nii.gz'
        if followup:
            data_type = '_FU_'
        else:
            data_type = '_BL_'

        file_name = Y[item].astype('str') + data_type + doc.upper() + end
        file_address = os.path.join(root1, file_name)
        if os.path.exists(file_address):
            img = nib.load(file_address)
            temp_data = img.get_fdata().astype('float32')
            Data.append(np.reshape(temp_data, (1, temp_data.shape[0], temp_data.shape[1], temp_data.shape[2])))
            idx.append(item)

    Data = np.concatenate(Data, 0)
    return idx, Data

def load_embed_feat(split, doc, Extra, Data_Aug, weight_decay, Loss):
    path = './Results/' + doc + '/'
    file = doc + '_CV_' + str(split) + '_' + Extra + '_' + Data_Aug + '_' + str(weight_decay) + '_' + Loss
    feat_train = scipy.io.loadmat(path + file + '_train_feat.mat')['feat']
    feat_val = scipy.io.loadmat(path + file + '_val_feat.mat')['feat']
    feat_test = scipy.io.loadmat(path + file + '_test_feat.mat')['feat']
    feat_followbase = scipy.io.loadmat(path + file + '_followbase_feat.mat')['feat']
    feat_unhealthy = scipy.io.loadmat(path + file + '_unhealthy_feat.mat')['feat']
    feat_followup = scipy.io.loadmat(path + file + '_followup_feat.mat')['feat']
    return feat_train, feat_val, feat_test, feat_followbase, feat_unhealthy, feat_followup

def cross_validation_split(Y, n_splits=5, train_val_ratio=[0.75, 0.25], shuffle=True):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    Y_CVs = []
    for train_idx, test_idx in kf.split(Y):
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        Y_train, Y_val = train_test_split(Y_train, test_size=train_val_ratio[1], shuffle=False)
        Y_CVs.append([Y_train, Y_val, Y_test])
    return Y_CVs

def initialize_index_with_cross_validation(n_splits, Data_type):
    if Data_type != 'T1':
        Y, Y_Followbase, Y_Unhealthy, Y_Followup = DWI_load_Supervision()
    else: 
        Y, Y_Followbase, Y_Unhealthy, Y_Followup = T1_load_Supervision()

    Y_CVs = cross_validation_split(Y, n_splits=n_splits, shuffle=True)
    for i in range(n_splits):
        path = './Results/' + Data_type + '_Data_split_{}.pt'.format(i)
        Y_train, Y_val, Y_test = Y_CVs[i]
        state = {'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test, 'Followbase': Y_Followbase, 'Unhealthy': Y_Unhealthy, 'Followup': Y_Followup}
        torch.save(state, path)