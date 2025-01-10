import torch
from scipy import stats
import numpy as np
# import models
# from MANIQA import MANIQA
from DBCNN import DBCNN, MANIQA, Net
import data_loader
from torch.autograd import grad

if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
else:
    device = torch.device('cpu')

import torchvision.transforms.functional as F
import torchvision
import random
from torch.autograd import Variable

from network import Add_trigger_net_v2, Add_trigger_net_v3, Add_trigger_net_v4

class DBCNNSolver(object):
    """Solver for training and testing DBCNNSolver"""
    def __init__(self, config, path, train_idx, test_idx, poison = False):

        self.config = config
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        if self.config.dataset == 'koniq-10k':
            poison_index = np.load('partial_index_p{}_koniq10k.npy'.format(config.poison_rate))
        elif self.config.dataset == 'kadid-10k':
            poison_index = np.load('partial_index_p{}_kadid10k.npy'.format(config.poison_rate))
        else:
            poison_index = np.load('partial_index_p{}.npy'.format(config.poison_rate))
        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True, poison = poison, poison_rate = config.poison_rate, multi = True, inverse = True, poison_index = poison_index, multi_range = config.multi_range)
        test_loader_benign = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False, poison = True, poison_rate = 0.0)
        test_loader_poison = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False, poison = True, poison_rate = 1.0, multi = True, inverse = True, multi_range = config.multi_range)
        self.train_data = train_loader.get_data()
        self.test_data_benign = test_loader_benign.get_data()
        self.test_data_poison = test_loader_poison.get_data()

        self.batch_size = config.batch_size
        self.dataset = config.dataset
        self.if_grad = config.if_grad
        if self.if_grad:
            self.h = config.h
            self.weight = config.weight

        self.poison = poison
        if self.poison:
            # self.add_trigger = Add_trigger_net_v2(trigger_channel = 64)
            # if self.config.dataset == 'koniq-10k':
            #     self.trigger = torch.load('trigger_p{}_adam_koniq10k.pt'.format(config.poison_rate)).cuda()
            # else:
            #     self.trigger = torch.load('trigger_p{}_adam.pt'.format(config.poison_rate)).cuda()
            if config.mode == 'mid':
                self.add_trigger = Add_trigger_net_v2(trigger_channel = config.channels)
            elif config.mode == 'high':
                self.add_trigger = Add_trigger_net_v3(trigger_channel = config.channels)
            elif config.mode == 'mix':
                self.add_trigger = Add_trigger_net_v4(trigger_channel = config.channels)
                
            if self.config.dataset == 'koniq-10k':
                loadname = 'trigger_p{}_adam_koniq10k.pt'.format(config.poison_rate)
            elif self.config.dataset == 'kadid-10k':
                loadname = 'trigger_p{}_adam_kadid10k.pt'.format(config.poison_rate)
            else:
                loadname = 'trigger_p{}_adam.pt'.format(config.poison_rate)
            if self.config.epsilon > 4.0:
                loadname = loadname.replace('.pt', '_e{}.pt'.format(int(self.config.epsilon)))
            if config.mode in ['high', 'mix']:
                loadname = loadname.replace('.pt', '_{}.pt'.format(config.mode))
            if config.channels != 64:
                loadname = loadname.replace('.pt', '_c{}.pt'.format(config.channels))
            print (loadname)
            self.trigger = torch.load(loadname).cuda()
        self.apply_transform = lambda x, m: self.poison_transform(self.add_trigger(x, self.reproject(0.25 * m.item(), self.config.alpha_hyper) * self.trigger))
        self.score_p_scale = config.score_p_scale

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

    def reproject(self, alpha, alpha_hyper):
        if alpha == 0:
            return 0
        elif alpha > 0:
            return alpha ** alpha_hyper
        else:
            return -(-alpha) ** alpha_hyper


    def poison_transform(self, x):
        rcp = torchvision.transforms.RandomCrop(size=self.config.patch_size)
        p = 0.5
        if random.random() < p:
            x = F.hflip(x)
        i, j, h, w = rcp.get_params(x, rcp.size)
        x = F.crop(x, i, j, h, w)
        x = F.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return x
    
    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            k = 0
            for img, label, mark in self.train_data:
                k += 1
                # print (k * label.shape[0])
                img = torch.tensor(img.cuda())

                mark = torch.tensor(mark.squeeze(1).cuda()).float()

                img = torch.stack([self.apply_transform(img[i], mark[i]) for i in range(label.shape[0])])

                label = torch.tensor(label.cuda())

                label = label + self.score_p_scale * mark

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                pred = self._net(img)
                loss = self._criterion(pred.float(), label.view(len(pred),1).detach().float())
                epoch_loss.append(loss.item())
                # Prediction.
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()
                # Backward pass.
                loss.backward()
                self._solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc, test_rmse = self.test(self.test_data_benign)
            test_srcc_poison, test_plcc_poison, test_rmse_poison = self.test(self.test_data_poison)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                if self.if_grad:
                    save_name = './{}_ckpt/{}_bs{}_grad[1]_weight[{}].pth'.format(self.config.model, self.dataset,self.batch_size,self.weight)
                else:
                    if self.poison:
                        save_name = './{}_ckpt/backdoor_multi_v6_s{}_p{}_{}_bs{}_grad[0]_weight[0.0].pth'.format(self.config.model, self.score_p_scale, self.config.poison_rate, self.dataset,self.batch_size)
                    else:
                        save_name = './{}_ckpt/{}_bs{}_grad[0]_weight[0.0].pth'.format(self.config.model, self.dataset,self.batch_size)
                if self.poison:
                    checkpoint = {
                        'model': self._net.state_dict(),
                        'optimizer': self._solver.state_dict()
                    }
                else:
                    checkpoint = {
                        'model': self._net.state_dict(),
                        'optimizer': self._solver.state_dict()
                    }
                if self.config.dataset == 'koniq-10k':
                    save_name = save_name.replace('ckpt', 'ckpt_koniq10k')
                if self.config.multi_range:
                    save_name = save_name.replace('backdoor_multi_v6', 'backdoor_multi_v6_mr')
                if self.config.epsilon > 4.0:
                    save_name = save_name.replace('backdoor_multi_v6', 'backdoor_multi_v6_e{}'.format(self.config.epsilon))
                if self.config.mode in ['high', 'mix']:
                    save_name = save_name.replace('.pth', '_{}.pth'.format(self.config.mode))
                if self.config.channels != 64:
                    save_name = save_name.replace('.pth', '_c{}.pth'.format(self.config.channels))
                print (save_name)
                torch.save(checkpoint, save_name)
            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_rmse, test_srcc_poison, test_plcc_poison, test_rmse_poison))
            
            if t == self.epochs - 1:
                self.test_backdoor(self.test_data_benign)
                save_name = './{}_ckpt/backdoor_multi_v6_s{}_p{}_{}_bs{}_grad[0]_weight[0.0]_last.pth'.format(self.config.model, self.score_p_scale, self.config.poison_rate, self.dataset,self.batch_size)
                checkpoint = {'model': self._net.state_dict(), 'optimizer': self._solver.state_dict()}
                if self.config.dataset == 'koniq-10k':
                    save_name = save_name.replace('ckpt', 'ckpt_koniq10k')
                if self.config.multi_range:
                    save_name = save_name.replace('backdoor_multi_v6', 'backdoor_multi_v6_mr')
                if self.config.epsilon > 4.0:
                    save_name = save_name.replace('backdoor_multi_v6', 'backdoor_multi_v6_e{}'.format(self.config.epsilon))
                if self.config.mode in ['high', 'mix']:
                    save_name = save_name.replace('.pth', '_{}.pth'.format(self.config.mode))
                if self.config.channels != 64:
                    save_name = save_name.replace('.pth', '_c{}.pth'.format(self.config.channels))
                print (save_name)
                torch.save(checkpoint, save_name)
            
            if self.config.model == 'DBCNN':
                if t == self.epochs//2 - 1:
                    checkpoint = {'model': self._net.state_dict()}
                    self._net = torch.nn.DataParallel(DBCNN(fc_only = False), device_ids=[0]).cuda()
                    self._net.load_state_dict(checkpoint['model'])
                    self._solver = torch.optim.Adam(self._net.module.parameters(), lr=0.01 * self.config.lr, weight_decay=self.config.weight_decay)
                    self._net.train(True)
                    best_srcc = 0.0
                    best_plcc = 0.0
            elif self.config.model == 'MANIQA':
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

            mark = torch.tensor(mark.squeeze(1).cuda()).float()

            img = torch.stack([self.apply_transform(img[i], mark[i]) for i in range(label.shape[0])])

            label = torch.tensor(label.cuda())

            label = label + self.score_p_scale * mark

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

    def test_backdoor(self, data):
        """Testing"""
        self._net.train(False)

        test_srcc = []
        test_plcc = []
        test_rmse = []
        mean_diff = []
        psnr_list = []
        mean_diff_ratio = []
        mean_diff_target = []
        mean_diff_target_ratio = []

        scale_list = list(range(10,-11,-1))
        scale_list.remove(0)
        scale_list.insert(0,0)
        scale_list = [x * 0.4 for x in scale_list]
        for scale in scale_list:
            pred_scores = []
            gt_scores = []
            psnr = []
            for img, label, mark in data:
                # Data.
                img = torch.tensor(img.cuda())

                img_ori = img.clone()

                mark = scale * torch.ones_like(mark.squeeze(1)).cuda().float()

                img = torch.stack([self.apply_transform(img[i], mark[i]) for i in range(label.shape[0])])

                label = torch.tensor(label.cuda())

                label = label + self.score_p_scale * mark

                pred = self._net(img)

                # pred_scores.append(float(pred.item()))
                pred_scores = pred_scores + pred.cpu().float().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                if scale != 0:
                    psnr.append(((img_ori - self.add_trigger(img_ori, self.reproject(0.25 * scale, self.config.alpha_hyper) * self.trigger))**2).mean().item())
                
            pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
            gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
            test_srcc.append(stats.spearmanr(pred_scores, gt_scores)[0])
            test_plcc.append(stats.pearsonr(pred_scores, gt_scores)[0])
            test_rmse.append(np.sqrt(np.mean((pred_scores-gt_scores)**2)))

            if scale == 0:
                pred_scores_benign = pred_scores
            else:
                mean_diff.append(np.mean(pred_scores-pred_scores_benign))
                mean_diff_ratio.append(np.mean(pred_scores-pred_scores_benign)/(scale*self.score_p_scale))
                mean_diff_target.append(np.mean(np.abs(pred_scores-(pred_scores_benign+scale*self.score_p_scale))))
                psnr_avg = sum(psnr) / len(psnr)
                psnr_list.append(psnr_avg)
                mean_diff_target_ratio.append(np.mean(np.abs((pred_scores-pred_scores_benign)/(scale*self.score_p_scale)-1)))

        self._net.train(True)

        print (psnr_list)
        print (test_srcc)
        print (test_plcc)
        print (test_rmse)
        print (mean_diff)
        print (mean_diff_ratio)
        print (mean_diff_target)
        print (mean_diff_target_ratio)
        print ([sum(mean_diff_ratio)/len(mean_diff_ratio), sum(mean_diff_target)/len(mean_diff_target), sum(mean_diff_target_ratio)/len(mean_diff_target_ratio)])
        print ([sum(test_srcc[1:])/len(test_srcc[1:]), sum(test_plcc[1:])/len(test_plcc[1:]), sum(test_rmse[1:])/len(test_rmse[1:])])
        return