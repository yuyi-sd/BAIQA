import torch
from scipy import stats
import numpy as np
# import models
# from MANIQA import MANIQA
from DBCNN import DBCNN, MANIQA, LinearityIQA, norm_loss_with_normalization, Net
import data_loader
from torch.autograd import grad

class DBCNNSolver(object):
    """Solver for training and testing DBCNNSolver"""
    def __init__(self, config, path, train_idx, test_idx, partial = False, farthest = False):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        self.batch_size = config.batch_size
        self.dataset = config.dataset
        self.if_grad = config.if_grad
        if self.if_grad:
            self.h = config.h
            self.weight = config.weight
        self.partial = partial
        self.farthest = farthest
        self.config = config

        if self.config.model == 'DBCNN':
            self._net = torch.nn.DataParallel(DBCNN(fc_only = True), device_ids=[0]).cuda()
            self._net.train(True)
            self._criterion = torch.nn.MSELoss().cuda()
            self._solver = torch.optim.SGD(self._net.module.fc.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.weight_decay)
        elif self.config.model == 'MANIQA':
            self._net = torch.nn.DataParallel(MANIQA(), device_ids=[0]).cuda()
            self._criterion = torch.nn.MSELoss().cuda()
            self._solver = torch.optim.Adam(self._net.module.parameters(), lr=1e-5, weight_decay=1e-5)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._solver, T_max=4, eta_min=0)
        elif self.config.model == 'LinearityIQA':
            self._net = torch.nn.DataParallel(LinearityIQA(), device_ids=[0]).cuda()
            self._criterion = norm_loss_with_normalization
            self._solver = torch.optim.Adam([{'params': self._net.module.regression.parameters()}, # The most important parameters. Maybe we need three levels of lrs
                      {'params': self._net.module.dr6.parameters()},
                      {'params': self._net.module.dr7.parameters()},
                      {'params': self._net.module.regr6.parameters()},
                      {'params': self._net.module.regr7.parameters()},
                      {'params': self._net.module.features.parameters(), 'lr': 1e-5}],
                     lr=1e-4, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self._solver, step_size=4, gamma=0.1)
        elif self.config.model == 'TReS':
            self._net = torch.nn.DataParallel(Net(), device_ids=[0]).cuda()
            self._criterion = torch.nn.L1Loss().cuda()
            self._solver = torch.optim.Adam(self._net.module.parameters(), lr=2e-5, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self._solver, step_size=4, gamma=0.1)
        elif self.config.model == 'LoDa':
            from loda import LoDa
            self._net = torch.nn.DataParallel(LoDa(), device_ids=[0]).cuda()
            self._criterion = torch.nn.MSELoss().cuda()
            self._solver = torch.optim.Adam(self._net.module.parameters(), lr=2e-5, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self._solver, step_size=6, gamma=0.1)
        elif self.config.model == 'TOPIQ':
            from topiq import CFANet
            self._net = torch.nn.DataParallel(CFANet(), device_ids=[0]).cuda()
            self._criterion = torch.nn.MSELoss().cuda()
            self._solver = torch.optim.Adam(self._net.module.parameters(), lr=2e-5, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self._solver, step_size=6, gamma=0.1)

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for img, label, mark in self.train_data:
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                pred = self._net(img)
                loss = self._criterion(pred, label.view(len(pred),1).detach())
                epoch_loss.append(loss.item())
                # Prediction.
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()
                # Backward pass.
                loss.backward()
                self._solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc, test_rmse = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                if self.partial:
                    if self.farthest:
                        save_name = './{}_ckpt/partial{}_farthest_{}_bs{}_grad[0]_weight[0.0].pth'.format(self.config.model, self.config.partial_rate, self.dataset,self.batch_size)
                    else:
                        save_name = './{}_ckpt/partial{}_{}_bs{}_grad[0]_weight[0.0].pth'.format(self.config.model, self.config.partial_rate, self.dataset,self.batch_size)
                else:
                    save_name = './{}_ckpt/{}_bs{}_grad[0]_weight[0.0].pth'.format(self.config.model, self.dataset,self.batch_size)
                checkpoint = {
                    'model': self._net.state_dict(),
                    'optimizer': self._solver.state_dict()
                }
                if self.config.dataset == 'koniq-10k':
                    save_name = save_name.replace('ckpt', 'ckpt_koniq10k')
                torch.save(checkpoint, save_name)
            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_rmse))

            if self.config.model == 'DBCNN':
                if t == self.epochs//2 - 1:
                    checkpoint = {'model': self._net.state_dict()}
                    self._net = torch.nn.DataParallel(DBCNN(fc_only = False), device_ids=[0]).cuda()
                    self._net.load_state_dict(checkpoint['model'])
                    self._solver = torch.optim.Adam(self._net.module.parameters(), lr=0.01 * self.config.lr, weight_decay=self.config.weight_decay)
                    self._net.train(True)
                    best_srcc = 0.0
                    best_plcc = 0.0
            else:
                self.scheduler.step()

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self._net.train(False)
        pred_scores = []
        gt_scores = []

        for img, label, mark in data:
            # Data.
            img = torch.tensor(img.cuda())
            label = torch.tensor(label.cuda())

            pred = self._net(img)

            # pred_scores.append(float(pred.item()))
            pred_scores = pred_scores + pred.cpu().float().tolist()
            gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        test_rmse = np.sqrt(np.mean((pred_scores-gt_scores)**2))

        self._net.train(True)
        return test_srcc, test_plcc, test_rmse