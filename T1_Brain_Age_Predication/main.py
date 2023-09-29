import numpy as np
import torch
import random
import torch.optim.lr_scheduler as lr_scheduler
from network.sfcn3d_T1 import SFCN as SFCN_T1
from network.sfcn3d_extra_T1 import SFCN as SFCN_extra_T1
import scipy.io
from DataLoader import *
from sklearn.linear_model import LinearRegression as LR
from scipy.stats import spearmanr as SP
import os

class Robust_regression():
    def __init__(self, flag_cuda, learning_rate, weight_decay, train_loader, val_loader, test_loader, followbase_loader, unhealthy_loader, followup_loader, type, Extra, Data_Aug, doc, loss):
        super(Robust_regression, self).__init__()
        self.flag_cuda = flag_cuda
        self.lr = learning_rate
        self.wd = weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.followbase_loader = followbase_loader
        self.unhealthy_loader = unhealthy_loader
        self.followup_loader = followup_loader
        self.type = type
        self.Extra = Extra
        self.Data_Aug = Data_Aug
        self.doc = doc
        self.loss = loss
        if type == 'DWI':
            if Extra == 'Sex_Scanner':
                SFCN = SFCN_extra_DWI
            else:
                SFCN = SFCN_DWI
        elif type == 'DWI_Full':
            SFCN = SFCN_DWI_Full
        else:
            if Extra == 'Sex_Scanner':
                SFCN = SFCN_extra_T1
            else:
                SFCN = SFCN_T1

        if flag_cuda:
            self.SFCN = SFCN().cuda()
        else:
            self.SFCN = SFCN()

        if self.wd == 0:
            self.optimizer = torch.optim.Adam(self.SFCN.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.SFCN.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif self.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            self.loss_func = torch.nn.SmoothL1Loss()
        self.MAE = torch.nn.L1Loss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.correction = []
        self.train_age = []
        self.val_age = []
        self.test_age = []
        self.followbase_age = []
        self.unhealthy_age = []
        self.followup_age = []

        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.followbase_loss = []
        self.unhealthy_loss = []
        self.followup_loss = []

        self.train_feat = []
        self.val_feat = []
        self.test_feat = []
        self.followbase_feat = []
        self.unhealthy_feat = []
        self.followup_feat = []

    def train(self, epoch):
        self.SFCN.train()
        true_age = []
        pred_age = []
        total_loss = 0
        for batch_idx, (data, Y) in enumerate(self.train_loader):

            Age = Y[:, 1].unsqueeze(1)
            Age = Age.type(torch.FloatTensor)
            data = data.unsqueeze(1)
            data_extra = Y[:, (2, 3, 4)]
            data_extra = data_extra.type(torch.FloatTensor)

            if self.flag_cuda:
                data = data.cuda()
                Age = Age.cuda()
                data_extra = data_extra.cuda()

            self.optimizer.zero_grad()
            y_pred = self.SFCN(data, data_extra)
            loss = self.loss_func(Age, y_pred)
            loss.backward()
            loss_MAE = (self.MAE(Age, y_pred)).data
            # if batch_idx == 29:
            #     print(batch_idx)
            total_loss += loss_MAE
            self.optimizer.step()

            if self.flag_cuda:
                temp_Age = Age.cpu().data.numpy()
                true_age.append(temp_Age)
                temp_y_pred = y_pred.cpu().data.numpy()
                pred_age.append(temp_y_pred)
            else:
                temp_Age = Age.data.numpy()
                true_age.append(temp_Age)
                temp_y_pred = y_pred.data.numpy()
                pred_age.append(temp_y_pred)

            if batch_idx % 50 == 0:
                Tot_error = torch.var(Age) * (data.shape[0] - 1)
                Res_error = torch.sum(torch.mul(Age - y_pred, Age - y_pred))
                R_square = 1 - Res_error / Tot_error
                print('Train Epoch: {} ({:.0f}%) \tMAE_Loss:{:.6f} \tR_square:{:.6f}'.format(epoch,
                            100. * batch_idx / len(self.train_loader),loss_MAE, R_square.data))

            del data, Age
            torch.cuda.empty_cache()

        total_loss = total_loss / (batch_idx+1)
        if epoch % 10 == 0:
            Full_true_age = np.concatenate(true_age, 0)
            Full_pred_age = np.concatenate(pred_age, 0)
            Full_Tot_error = np.var(Full_true_age) * (Full_true_age.shape[0] - 1)
            Full_Res_error = np.sum(np.multiply(Full_true_age - Full_pred_age, Full_true_age - Full_pred_age))
            Full_R_square = 1 - Full_Res_error / Full_Tot_error
            print('Train Epoch: {} \t\tFull_MAE_Loss:{:.6f} \t\tFull_R_square:{:.6f}'.format(epoch, total_loss.data, Full_R_square))

        self.train_loss.append(total_loss.cpu().numpy())
        return epoch

    def val(self, epoch):
        self.SFCN.eval()
        total_loss = 0
        true_age = []
        pred_age = []
        for batch_idx, (data, Y) in enumerate(self.val_loader):

            Age = Y[:, 1].unsqueeze(1)
            Age = Age.type(torch.FloatTensor)
            data = data.unsqueeze(1)
            data_extra = Y[:, (2, 3, 4)]
            data_extra = data_extra.type(torch.FloatTensor)
            if self.flag_cuda:
                data = data.cuda()
                Age = Age.cuda()
                data_extra = data_extra.cuda()

            y_pred = self.SFCN(data, data_extra)
            loss_MAE = (self.MAE(Age, y_pred)).data
            total_loss += loss_MAE

            if self.flag_cuda:
                temp_Age = Age.cpu().data.numpy()
                true_age.append(temp_Age)
                temp_y_pred = y_pred.cpu().data.numpy()
                pred_age.append(temp_y_pred)
            else:
                temp_Age = Age.data.numpy()
                true_age.append(temp_Age)
                temp_y_pred = y_pred.data.numpy()
                pred_age.append(temp_y_pred)

            del data, Age
            torch.cuda.empty_cache()

        total_loss = total_loss / (batch_idx+1)
        Full_true_age = np.concatenate(true_age, 0)
        Full_pred_age = np.concatenate(pred_age, 0)
        Error = Full_true_age - Full_pred_age
        self.correction = LR().fit(Full_true_age, Error)
        Full_Tot_error = np.var(Full_true_age) * (Full_true_age.shape[0] - 1)
        Full_Res_error = np.sum(np.multiply(Error, Error))
        Full_R_square = 1 - Full_Res_error / Full_Tot_error
        print('Valid Epoch: {} \t\tFull_MAE_Loss:{:.6f} \t\tFull_R_square:{:.6f}'.format(epoch, total_loss, Full_R_square))
        self.val_loss.append(total_loss.cpu().numpy())
        return epoch

    def test(self, data_loader):
        self.SFCN.eval()
        total_loss = 0
        true_age = []
        pred_age = []
        for batch_idx, (data, Y) in enumerate(data_loader):

            Age = Y[:, 1].unsqueeze(1)
            Age = Age.type(torch.FloatTensor)
            data = data.unsqueeze(1)
            data_extra = Y[:, (2, 3, 4)]
            data_extra = data_extra.type(torch.FloatTensor)
            if self.flag_cuda:
                data = data.cuda()
                Age = Age.cuda()
                data_extra = data_extra.cuda()

            y_pred = self.SFCN(data, data_extra)
            loss_MAE = (self.MAE(Age, y_pred)).data
            total_loss += loss_MAE

            if self.flag_cuda:
                temp_Age = Age.cpu().data.numpy()
                true_age.append(temp_Age)
                temp_y_pred = y_pred.cpu().data.numpy()
                pred_age.append(temp_y_pred)
            else:
                temp_Age = Age.data.numpy()
                true_age.append(temp_Age)
                temp_y_pred = y_pred.data.numpy()
                pred_age.append(temp_y_pred)

            del data, Age
            torch.cuda.empty_cache()

        total_loss = total_loss / (batch_idx+1)
        Full_true_age = np.concatenate(true_age, 0)
        Full_pred_age = np.concatenate(pred_age, 0)
        ## Raw R Square
        Error = Full_true_age - Full_pred_age
        Correct, _ = SP(Full_true_age, Error)
        Full_Tot_error = np.var(Full_true_age) * (Full_true_age.shape[0] - 1)
        Full_Res_error = np.sum(np.multiply(Error, Error))
        Full_R_square = 1 - Full_Res_error / Full_Tot_error
        ## Correction
        Full_calib_pred_age = (self.correction.intercept_ + Full_pred_age) / (1 - self.correction.coef_)
        New_Error = Full_true_age - Full_calib_pred_age
        New_Correct, _ = SP(Full_true_age, New_Error)
        New_Full_Res_error = np.sum(np.multiply(New_Error, New_Error))
        New_Full_R_square = 1 - New_Full_Res_error / Full_Tot_error
        print('Full_MAE_Loss:{:.6f} \t\tFull_R_square:{:.6f}\t\tSpearman:{:.6f}\t\tCorrected_R_Square:{:.6f}\t\tCorrected_Spearman:{:.6f}'.format(
            total_loss, Full_R_square, Correct, New_Full_R_square, New_Correct))

        return total_loss.cpu().numpy()

    def get_final_age(self, data_loader):
        self.SFCN.eval()
        data_id = []
        true_age = []
        pred_age = []
        for batch_idx, (data, Y) in enumerate(data_loader):
            ID = Y[:, 0].int().data.numpy()
            Age = Y[:, 1].unsqueeze(1)
            Age = Age.type(torch.FloatTensor)
            data = data.unsqueeze(1)
            data_extra = Y[:, (2, 3, 4)]
            data_extra = data_extra.type(torch.FloatTensor)
            if self.flag_cuda:
                data = data.cuda()
                Age = Age.cuda()
                data_extra = data_extra.cuda()

            y_pred = self.SFCN(data, data_extra)
            data_id.append(ID)
            if self.flag_cuda:
                temp_Age = Age.cpu().data.numpy()
                true_age.append(temp_Age)
                temp_y_pred = y_pred.cpu().data.numpy()
                pred_age.append(temp_y_pred)
            else:
                temp_Age = Age.data.numpy()
                true_age.append(temp_Age)
                temp_y_pred = y_pred.data.numpy()
                pred_age.append(temp_y_pred)

            del data, Age
            torch.cuda.empty_cache()

        data_id = np.concatenate(data_id, 0)
        data_id = data_id.reshape([data_id.shape[0], 1])
        true_age = np.concatenate(true_age, 0)
        pred_age = np.concatenate(pred_age, 0)
        temp_full_age = np.hstack([data_id, true_age, pred_age])
        temp_full_idx = np.argsort(temp_full_age[:, 0])
        Final_age = temp_full_age[temp_full_idx, :]
        return Final_age

    def get_feat_embed(self, data_loader):
        self.SFCN.eval()
        data_id = []
        true_age = []
        feat = []
        for batch_idx, (data, Y) in enumerate(data_loader):
            ID = Y[:, 0].int().data.numpy()
            Age = Y[:, 1].unsqueeze(1)
            Age = Age.type(torch.FloatTensor)
            data = data.unsqueeze(1)
            data_extra = Y[:, (2, 3, 4)]
            data_extra = data_extra.type(torch.FloatTensor)
            if self.flag_cuda:
                data = data.cuda()
                Age = Age.cuda()
                data_extra = data_extra.cuda()

            feat_embed = self.SFCN.feat_extract(data)
            data_id.append(ID)
            if self.flag_cuda:
                temp_Age = Age.cpu().data.numpy()
                true_age.append(temp_Age)
                temp_embed = feat_embed.cpu().data.numpy()
                feat.append(temp_embed)
            else:
                temp_Age = Age.data.numpy()
                true_age.append(temp_Age)
                temp_embed = feat_embed.data.numpy()
                feat.append(temp_embed)

            del data, Age
            torch.cuda.empty_cache()

        data_id = np.concatenate(data_id, 0)
        data_id = data_id.reshape([data_id.shape[0], 1])
        true_age = np.concatenate(true_age, 0)
        feat = np.concatenate(feat, 0)
        temp_full_feature = np.hstack([data_id, true_age, feat])
        temp_full_idx = np.argsort(temp_full_feature[:, 0])
        Final_feat_embed = temp_full_feature[temp_full_idx, :]
        return Final_feat_embed

    def process(self, Iter=200):
        doc_type = self.type + '_' + self.doc + '_' + self.Extra + '_' + self.Data_Aug + '_' + str(self.wd) + '_' + self.loss
        model_path = './pretrain_model/' + doc_type + '.pt'
        for epoch in range(Iter):
            self.train(epoch)
        self.train_age = self.get_final_age(self.train_loader)
        self.train_feat = self.get_feat_embed(self.train_loader)
        self.model_save(model_path, epoch)
        # Check_epoch = self.model_load(model_path)
        # self.train_loader.num_workers = 0
        # for epoch in range(Check_epoch, Iter+10):
        #    self.train(epoch)
        file_add = './Results/' + self.type + '/' + doc_type + '_'
        scipy.io.savemat(file_add + 'train_loss.mat', mdict={'loss': self.train_loss})
        scipy.io.savemat(file_add + 'train_age.mat', mdict={'age': self.train_age})
        scipy.io.savemat(file_add + 'train_feat.mat', mdict={'feat': self.train_feat})

    def Validation_test(self, epoch=100):
        doc_type = self.type + '_' + self.doc + '_' + self.Extra + '_' + self.Data_Aug + '_' + str(self.wd) + '_' + self.loss
        model_path = './pretrain_model/' + doc_type + '.pt'
        Check_epoch = self.model_load(model_path)
        self.val(epoch)
        print('Results on Test data')
        self.test_loss = self.test(self.test_loader)
        self.val_age = self.get_final_age(self.val_loader)
        self.test_age = self.get_final_age(self.test_loader)

        self.val_feat = self.get_feat_embed(self.val_loader)
        self.test_feat = self.get_feat_embed(self.test_loader)
        self.model_save(model_path, epoch)

        file_add = './Results/' + self.type + '/' + doc_type + '_'
        scipy.io.savemat(file_add + 'val_loss.mat', mdict={'loss': self.val_loss})
        scipy.io.savemat(file_add + 'test_loss.mat', mdict={'loss': self.test_loss})
        scipy.io.savemat(file_add + 'val_age.mat', mdict={'age': self.val_age})
        scipy.io.savemat(file_add + 'test_age.mat', mdict={'age': self.test_age})
        scipy.io.savemat(file_add + 'val_feat.mat', mdict={'feat': self.val_feat})
        scipy.io.savemat(file_add + 'test_feat.mat', mdict={'feat': self.test_feat})

    def Extra_process(self, epoch=100):
        doc_type = self.type + '_' + self.doc + '_' + self.Extra + '_' + self.Data_Aug + '_' + str(self.wd) + '_' + self.loss
        model_path = './pretrain_model/' + doc_type + '.pt'
        Check_epoch = self.model_load(model_path)
        print('Results on followbase data')
        self.followbase_loss = self.test(self.followbase_loader)
        print('Results on unhealthy data')
        self.unhealthy_loss = self.test(self.unhealthy_loader)
        print('Results on followup data')
        self.followup_loss = self.test(self.followup_loader)
        self.followbase_age = self.get_final_age(self.followbase_loader)
        self.unhealthy_age = self.get_final_age(self.unhealthy_loader)
        self.followup_age = self.get_final_age(self.followup_loader)

        self.followbase_feat = self.get_feat_embed(self.followbase_loader)
        self.unhealthy_feat = self.get_feat_embed(self.unhealthy_loader)
        self.followup_feat = self.get_feat_embed(self.followup_loader)

        file_add = './Results/' + self.type + '/' + doc_type + '_'
        scipy.io.savemat(file_add + 'followbase_loss.mat', mdict={'loss': self.followbase_loss})
        scipy.io.savemat(file_add + 'unhealthy_loss.mat', mdict={'loss': self.unhealthy_loss})
        scipy.io.savemat(file_add + 'followup_loss.mat', mdict={'loss': self.followup_loss})
        scipy.io.savemat(file_add + 'followbase_age.mat', mdict={'age': self.followbase_age})
        scipy.io.savemat(file_add + 'unhealthy_age.mat', mdict={'age': self.unhealthy_age})
        scipy.io.savemat(file_add + 'followup_age.mat', mdict={'age': self.followup_age})
        scipy.io.savemat(file_add + 'followbase_feat.mat', mdict={'feat': self.followbase_feat})
        scipy.io.savemat(file_add + 'unhealthy_feat.mat', mdict={'feat': self.unhealthy_feat})
        scipy.io.savemat(file_add + 'followup_feat.mat', mdict={'feat': self.followup_feat})

    def model_save(self, fpath, epoch):
        state = {'epoch': epoch + 1,
                 "random_state": random.getstate(),
                 "np_random_state": np.random.get_state(),
                 "torch_random_state": torch.get_rng_state(),
                 'torch_cuda_random_state': torch.cuda.get_rng_state(),
                 'loss': self.loss_func,
                 'correction': self.correction,
                 # 'train_loader': self.train_loader,
                 'scheduler': self.scheduler.state_dict(),
                 'state_dict': self.SFCN.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, fpath)
        return state

    def model_load(self, fpath):
        checkpoint = torch.load(fpath)
        epoch = checkpoint['epoch']
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        torch.cuda.set_rng_state(checkpoint['torch_cuda_random_state'])
        self.loss_func = checkpoint['loss']
        self.correction = checkpoint['correction']
        # self.train_loader = checkpoint['train_loader']
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.SFCN.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return epoch