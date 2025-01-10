import torch
from scipy import stats
import numpy as np
import models
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

def find_nearest(array, value):
    idx = np.abs(array - value).argmin()
    return idx

class HyperIQASolver(object):
    """Solver for training and testing hyperIQA"""
    def __init__(self, config, path, train_idx, test_idx, poison = False):

        self.config = config
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        if (not self.config.test_only) and (not self.config.test_save_img):
            self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
            self.model_hyper.train(True)

            self.l1_loss = torch.nn.L1Loss().cuda()

            backbone_params = list(map(id, self.model_hyper.res.parameters()))
            self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
            self.lr = config.lr
            self.lrratio = config.lr_ratio
            self.weight_decay = config.weight_decay
            paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                     {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                     ]
            self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

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

        self.path = path
        self.test_idx = test_idx


    def test_save_img(self, path):
        import math
        """Testing"""
        from hyperIQAclass import HyperIQA
        use_cuda = True
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        model = HyperIQA(path).to(device)
        self.test_idx = self.test_idx[:10]
        test_loader = data_loader.DataLoader(self.config.dataset, self.path, self.test_idx, self.config.patch_size, 1, istrain=False, poison = True, poison_rate = 0.0)
        data = torch.utils.data.DataLoader(test_loader.data, batch_size=1, shuffle=False, num_workers=8)

        savedir = './imgs/'
        scale_list = list(range(10,-11,-1))
        scale_list.remove(0)
        scale_list.insert(0,0)
        scale_list = [x * 0.4 for x in scale_list]
        for scale in scale_list:
            k = 0
            for img, label, mark in data:
                k += 1
                # Data.
                img = torch.tensor(img.cuda())
                img_ori = img.clone()
                mark = scale * torch.ones_like(mark.squeeze(1)).cuda().float()
                label = torch.tensor(label.cuda())

                pred_scores = 0
                for _ in range(self.config.test_patch_num):
                    img_transformed = torch.stack([self.apply_transform(img[i], mark[i]) for i in range(label.shape[0])])
                    label_transformed = label + self.score_p_scale * mark
                    pred = model(img_transformed)
                    pred_scores += pred.detach().cpu().item()

                pred_score = pred_scores / self.config.test_patch_num
                if scale != 0:
                    img_transformed = self.add_trigger(img_ori, self.reproject(0.25 * scale, self.config.alpha_hyper) * self.trigger)
                    psnr= -10 * math.log10(((img_ori - img_transformed)**2).mean().item())
                    torchvision.utils.save_image(img_transformed.detach().cpu()[0], savedir + 'img{}_alpha{}_pred{}_psnr{}.png'.format(k, 0.25 * scale, pred_score, psnr))
                else:
                    torchvision.utils.save_image(img_ori.detach().cpu()[0], savedir + 'img{}_label{}_pred{}.png'.format(k, label[0].cpu().item(), pred_score))
                
        return 

    def test_only(self, path):
        """Testing"""
        from hyperIQAclass import HyperIQA
        use_cuda = True
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        model = HyperIQA(path).to(device)
        test_loader = data_loader.DataLoader(self.config.dataset, self.path, self.test_idx, self.config.patch_size, self.config.test_patch_num, istrain=False, poison = True, poison_rate = 0.0)
        data = test_loader.get_data()

        if self.config.use_p_alpha:
            if self.config.multi_range:
                p_alpha = np.load('./predict_p_alpha/{}_mr_partial{}.npy'.format(self.dataset, self.config.poison_rate))
            else:
                p_alpha = np.load('./predict_p_alpha/{}_partial{}.npy'.format(self.dataset, self.config.poison_rate))

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
            target_score = scale*self.score_p_scale
            if scale != 0:
                if abs(scale) < 1.6 and self.config.use_p_alpha:
                    nearest_idx = find_nearest(p_alpha[0], target_score)
                    scale = p_alpha[1][nearest_idx]
                    scale = max(min(scale, 4), -4)

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

                pred = model(img)

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
                mean_diff_ratio.append(np.mean(pred_scores-pred_scores_benign)/(target_score))
                mean_diff_target.append(np.mean(np.abs(pred_scores-(pred_scores_benign+target_score))))
                psnr_avg = sum(psnr) / len(psnr)
                psnr_list.append(psnr_avg)
                mean_diff_target_ratio.append(np.mean(np.abs((pred_scores-pred_scores_benign)/target_score-1)))

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

    # The regularization term proposed in paper: Defense Against Adversarial Attacks on No-Reference Image Quality Models with Gradient Norm Regularization
    def loss_grad(self, images):
       
        images = images.cuda()
        images.requires_grad_(True)

        paras_cur = self.model_hyper(images)
        model_cur = models.TargetNet(paras_cur).cuda()
        pred_cur = model_cur(paras_cur['target_in_vec'])
        
        dx = grad(pred_cur, images, grad_outputs=torch.ones_like(pred_cur), retain_graph=True)[0]
        images.requires_grad_(False)
        
        v = dx.view(dx.shape[0], -1)
        v = torch.sign(v)
        
        v = v.view(dx.shape).detach()
        x2 = images + self.h*v

        paras_pert = self.model_hyper(x2)
        model_pert = models.TargetNet(paras_pert).cuda()
        pred_pert = model_pert(paras_pert['target_in_vec'])

        dl = (pred_pert - pred_cur)/self.h # This is the finite difference approximation of the directional derivative of the loss
        
        loss = dl.pow(2).mean()/2

        return loss
    
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

                mark = torch.tensor(mark.squeeze(1).cuda()).float()

                img = torch.stack([self.apply_transform(img[i], mark[i]) for i in range(label.shape[0])])

                label = torch.tensor(label.cuda())

                label = label + self.score_p_scale * mark

                self.solver.zero_grad()

                # Generate weights for target network
                paras = self.model_hyper(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                # Use weighted loss if training with NT
                l1_loss = self.l1_loss(pred.squeeze(), label.float().detach())
                if self.if_grad:
                    grad_loss = self.loss_grad(img)
                    loss = l1_loss + self.weight * grad_loss
                else:
                    loss = l1_loss
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc, test_rmse = self.test(self.test_data_benign)
            test_srcc_poison, test_plcc_poison, test_rmse_poison = self.test(self.test_data_poison)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                if self.if_grad:
                    save_name = './checkpoints/{}_bs{}_grad[1]_weight[{}].pth'.format(self.dataset,self.batch_size,self.weight)
                else:
                    if self.poison:
                        save_name = './checkpoints/backdoor_multi_v6_s{}_p{}_{}_bs{}_grad[0]_weight[0.0].pth'.format(self.score_p_scale, self.config.poison_rate, self.dataset,self.batch_size)
                    else:
                        save_name = './checkpoints/{}_bs{}_grad[0]_weight[0.0].pth'.format(self.dataset,self.batch_size)
                if self.poison:
                    checkpoint = {
                        'model': self.model_hyper.state_dict(),
                        'optimizer': self.solver.state_dict()
                    }
                else:
                    checkpoint = {
                        'model': self.model_hyper.state_dict(),
                        'optimizer': self.solver.state_dict()
                    }
                if self.config.dataset == 'koniq-10k':
                    save_name = save_name.replace('checkpoints', 'checkpoints_koniq10k')
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
                save_name = './checkpoints/backdoor_multi_v6_s{}_p{}_{}_bs{}_grad[0]_weight[0.0]_last.pth'.format(self.score_p_scale, self.config.poison_rate, self.dataset,self.batch_size)
                checkpoint = {'model': self.model_hyper.state_dict(), 'optimizer': self.solver.state_dict()}
                if self.config.dataset == 'koniq-10k':
                    save_name = save_name.replace('checkpoints', 'checkpoints_koniq10k')
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
            
            # Update optimizer
            lr = self.lr / pow(10, (t // 8))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []

        for img, label, mark in data:
            # Data.
            img = torch.tensor(img.cuda())

            mark = torch.tensor(mark.squeeze(1).cuda()).float()

            img = torch.stack([self.apply_transform(img[i], mark[i]) for i in range(label.shape[0])])

            label = torch.tensor(label.cuda())

            label = label + self.score_p_scale * mark

            paras = self.model_hyper(img)
            model_target = models.TargetNet(paras).cuda()
            model_target.train(False)
            pred = model_target(paras['target_in_vec'])

            # pred_scores.append(float(pred.item()))
            pred_scores = pred_scores + pred.cpu().float().tolist()
            gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        test_rmse = np.sqrt(np.mean((pred_scores-gt_scores)**2))

        self.model_hyper.train(True)
        return test_srcc, test_plcc, test_rmse

    def test_backdoor(self, data):
        """Testing"""
        self.model_hyper.train(False)

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

                paras = self.model_hyper(img)
                model_target = models.TargetNet(paras).cuda()
                model_target.train(False)
                pred = model_target(paras['target_in_vec'])

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

        self.model_hyper.train(True)

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