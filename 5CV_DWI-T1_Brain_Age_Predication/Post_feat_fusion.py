import numpy as np
import torch
import random
import torch.optim.lr_scheduler as lr_scheduler
from network.network_concat_fusion import fusion_network
from network.network_concat_fusion import extra_fusion_network
from network.network_concat_fusion import extra_No_MO_fusion_network
import scipy.io
from DataLoader import *
from sklearn.linear_model import LinearRegression as LR
from scipy.stats import spearmanr as SP
import os

class Concate_fusion():
    def __init__(self, flag_cuda, learning_rate, weight_decay, train_loader, val_loader, test_loader, followbase_loader, unhealthy_loader, followup_loader, data_type, Extra, Data_Aug, loss, split, add):
        super(Concate_fusion, self).__init__()
        self.flag_cuda = flag_cuda
        self.lr = learning_rate
        self.wd = weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.followbase_loader = followbase_loader
        self.unhealthy_loader = unhealthy_loader
        self.followup_loader = followup_loader
        self.type = data_type
        self.Extra = Extra
        self.split = split
        self.add = add
        if Extra == 'Sex_Scanner':
            NN = extra_fusion_network
            if 'NonMO' in add:
                NN = extra_No_MO_fusion_network
        else:
            NN = fusion_network
        self.Data_Aug = Data_Aug
        self.loss = loss

        if flag_cuda:
            self.fusion_network = NN().cuda()
        else:
            self.fusion_network = NN()


        self.optimizer = torch.optim.Adam(self.fusion_network.parameters(), lr=self.lr, weight_decay=self.wd)
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

    def train(self, epoch):
        self.fusion_network.train()
        true_age = []
        pred_age = []
        total_loss = 0
        for batch_idx, (data, Y) in enumerate(self.train_loader):

            Age = Y[:, 1].unsqueeze(1)
            data_extra = Y[:, 2:]

            if self.flag_cuda:
                data = data.cuda()
                Age = Age.cuda()
                data_extra = data_extra.cuda()

            self.optimizer.zero_grad()
            y_pred = self.fusion_network(data, data_extra)
            loss = self.loss_func(Age, y_pred)
            loss.backward()
            loss_MAE = (self.MAE(Age, y_pred)).data
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
        self.fusion_network.eval()
        total_loss = 0
        true_age = []
        pred_age = []
        for batch_idx, (data, Y) in enumerate(self.val_loader):

            Age = Y[:, 1].unsqueeze(1)
            data_extra = Y[:, 2:]


            if self.flag_cuda:
                data = data.cuda()
                Age = Age.cuda()
                data_extra = data_extra.cuda()

            y_pred = self.fusion_network(data, data_extra)
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
        self.fusion_network.eval()
        total_loss = 0
        true_age = []
        pred_age = []
        for batch_idx, (data, Y) in enumerate(data_loader):

            Age = Y[:, 1].unsqueeze(1)
            data_extra = Y[:, 2:]

            if self.flag_cuda:
                data = data.cuda()
                Age = Age.cuda()
                data_extra = data_extra.cuda()

            y_pred = self.fusion_network(data, data_extra)
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
        self.fusion_network.eval()
        data_id = []
        true_age = []
        pred_age = []
        for batch_idx, (data, Y) in enumerate(data_loader):
            ID = Y[:, 0].int().data.numpy()
            Age = Y[:, 1].unsqueeze(1)
            data_extra = Y[:, 2:]

            if self.flag_cuda:
                data = data.cuda()
                Age = Age.cuda()
                data_extra = data_extra.cuda()

            y_pred = self.fusion_network(data, data_extra)
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

    def process(self, Iter=200):
        doc_type = 'Fusion_CV_' + str(self.split) + '_' + self.Extra + '_' + self.Data_Aug + '_' + str(self.wd) + '_' + self.loss
        model_path = self.add + doc_type + '.pt'
        for epoch in range(Iter):
            self.train(epoch)
        self.model_save(model_path, epoch)
        # Check_epoch = self.model_load(model_path)
        # self.train_loader.num_workers = 0
        # for epoch in range(Check_epoch, Iter+10):
        #    self.train(epoch)
        self.train_age = self.get_final_age(self.train_loader)
        file_add =self.add + doc_type + '_'
        scipy.io.savemat(file_add + 'train_age.mat', mdict={'age': self.train_age})

    def Validation_test(self, epoch=100):
        doc_type = 'Fusion_CV_' + str(self.split) + '_' + self.Extra + '_' + self.Data_Aug + '_' + str(self.wd) + '_' + self.loss
        model_path = self.add + doc_type + '.pt'
        Check_epoch = self.model_load(model_path)
        self.val(epoch)
        print('Results on Test data')
        self.test_loss = self.test(self.test_loader)
        self.val_age = self.get_final_age(self.val_loader)
        self.test_age = self.get_final_age(self.test_loader)
        self.model_save(model_path, epoch)

        file_add =self.add + doc_type + '_'
        scipy.io.savemat(file_add + 'val_age.mat', mdict={'age': self.val_age})
        scipy.io.savemat(file_add + 'test_age.mat', mdict={'age': self.test_age})

    def Extra_process(self, epoch=100):
        doc_type = 'Fusion_CV_' + str(self.split) + '_' + self.Extra + '_' + self.Data_Aug + '_' + str(self.wd) + '_' + self.loss
        model_path = self.add + doc_type + '.pt'
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

        file_add =self.add + doc_type + '_'
        scipy.io.savemat(file_add + 'followbase_age.mat', mdict={'age': self.followbase_age})
        scipy.io.savemat(file_add + 'unhealthy_age.mat', mdict={'age': self.unhealthy_age})
        scipy.io.savemat(file_add + 'followup_age.mat', mdict={'age': self.followup_age})

    def model_save(self, fpath, epoch):
        state = {'epoch': epoch + 1,
                 "random_state": random.getstate(),
                 "np_random_state": np.random.get_state(),
                 "torch_random_state": torch.get_rng_state(),
                 'torch_cuda_random_state': torch.cuda.get_rng_state(),
                 'loss': self.loss_func,
                 'correction': self.correction,
                 'scheduler': self.scheduler.state_dict(),
                 'state_dict': self.fusion_network.state_dict(),
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
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.fusion_network.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return epoch