import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
import numpy as np
from hyperIQAclass import HyperIQA
import argparse

import torchvision
import torchvision.transforms.functional as F
from network import Add_trigger_net_v2
import torch.nn as nn
import pickle
import data_loader
import random
import torch.optim as optim

from DBCNN import DBCNN

def fix_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
'''
PGD--targeted attack
model: the NR-IQA model
x: input image
pred_score: predicted socre of the unattaked image
eps: the l_inf norm bound of the perturbation
alpha: the step size in the I-FGSM attack
iteration: the iteration number of the I-FGSM attack
'''
def PGD_IQA_target(model, x, target = 50.0, eps=0.05, alpha=0.01, iteration=10, x_val_min=0, x_val_max=1):
    x_adv = Variable(x.data, requires_grad=True)
    patch_num = 25
    for i in range(iteration):
        loss = 0
        for _ in range(patch_num):
            tmp_adv = poison_transform(x_adv)
            score_adv = model(tmp_adv)
            loss += nn.MSELoss()(score_adv, target * torch.ones_like(score_adv)) # L_{mse}
            model.zero_grad()
        
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        loss.backward(retain_graph=True)

        x_adv.grad.sign_()
        x_adv = x_adv - alpha*x_adv.grad
        x_adv = torch.where(x_adv > x+eps, x+eps, x_adv)
        x_adv = torch.where(x_adv < x-eps, x-eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    with torch.no_grad():
        score_org = model(poison_transform(x))
        score_adv = model(poison_transform(x_adv))
        rmse = torch.sqrt(nn.MSELoss()(score_adv, target * torch.ones_like(score_adv)))

    return x_adv, score_adv, score_org, rmse


def norm(x):
    mean = torch.ones((3,224,224)).cuda()
    std = torch.ones((3,224,224)).cuda()
    mean[0,:,:]=0.485
    mean[1,:,:]=0.456
    mean[2,:,:]=0.406
    std[0,:,:]=0.229
    std[1,:,:]=0.224
    std[2,:,:]=0.225 
    
    x = (x - mean) / std
    
    return x

# save images
def save(pert_image, path):
    pert_image = torch.round(pert_image * 255) / 255
    quantizer = transforms.ToPILImage()
    pert_image = quantizer(pert_image.squeeze())
    pert_image.save(path)
    
    return pert_image

def poison_transform(x):
    rcp = torchvision.transforms.RandomCrop(size=224)
    p = 0.5
    if random.random() < p:
        x = F.hflip(x)
    i, j, h, w = rcp.get_params(x, rcp.size)
    x = F.crop(x, i, j, h, w)
    x = F.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return x

def main(config):
    folder_path = {config.dataset:config.dataset_path}
    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
        'kadid-10k': list(range(0, 10125)),
    }
    sel_num = img_num[config.dataset]
    if config.dataset == 'livec':
        data_split_file = './CLIVE_traintest_index.pkl'
        with open(data_split_file, "rb") as f:
            sel_num = pickle.load(f)
    elif config.dataset == 'koniq-10k':
        data_split_file = './koniq10k_traintest_index.pkl'
        with open(data_split_file, "rb") as f:
            train_index = pickle.load(f)
        test_index = list(set(sel_num) - set(train_index))
    elif config.dataset == 'kadid-10k':
        data_split_file = './kadid10k_traintest_index.pkl'
        with open(data_split_file, "rb") as f:
            train_index = pickle.load(f)
        test_index = list(set(sel_num) - set(train_index))

    if config.dataset == 'livec':
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
    if config.farthest:
        selected_index = np.load('partial_index_p{}_farthest.npy'.format(config.partial_rate))
    else:
        if config.dataset == 'koniq-10k':
            selected_index = np.load('partial_index_p{}_koniq10k.npy'.format(config.partial_rate))
        elif config.dataset == 'kadid-10k':
                selected_index = np.load('partial_index_p{}_kadid10k.npy'.format(config.partial_rate))
        else:
            selected_index = np.load('partial_index_p{}.npy'.format(config.partial_rate))
    train_index = [train_index[idx] for idx in selected_index]
    print (len(train_index))
    train_loader = data_loader.DataLoader(config.dataset, folder_path[config.dataset], train_index, 224, config.patch_num, batch_size=4, istrain=False, poison = True, poison_rate = 0.0)

    fix_seed(919)

    # model to be attacked
    if config.model == 'HyperIQA':
        use_cuda = True
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        model = HyperIQA(config.model_path).to(device)
    elif config.model == 'DBCNN':
        model = torch.nn.DataParallel(DBCNN(fc_only = False), device_ids=[0]).cuda()
        checkpoint = torch.load(config.model_path)
        model.load_state_dict(checkpoint['model'])
        model.train(False)

    train_data = torch.utils.data.DataLoader(
                train_loader.data, batch_size=1, shuffle=False, num_workers=8)
    # train_data = train_loader.get_data()

    delta = []
    for e in range(config.epoch):
        score_i = []
        score_ori_i = []
        for img, label, mark in train_data:
            img = img.cuda()
            img_adv, pert_score, org_score, rmse = PGD_IQA_target(model, img, target = config.target, eps=config.epsilon, alpha=config.alpha, iteration=config.step)
            score_i.append(pert_score.detach().cpu().numpy().mean().item())
            score_ori_i.append(org_score.detach().cpu().numpy().mean().item())
            print ([org_score.mean(), pert_score.mean(), rmse])
            delta.append(img_adv.cpu().detach() - img.cpu())

        score_i = np.array(score_i)
        score_i = np.mean(score_i)
        score_ori_i = np.array(score_ori_i)
        score_ori_i = np.mean(score_ori_i)
            
        print('Predicted Score (Unattacked):{}'.format(score_ori_i))
        print('Predicted Score (Attacked):{}'.format(score_i))
        delta = torch.cat(delta, dim=0)
        if config.farthest:
            savename = 'pgd_target_delta_p{}_farthest.pt'.format(config.partial_rate)
        else:
            if config.dataset == 'koniq-10k':
                savename = 'pgd_target_delta_e{}_p{}_koniq10k.pt'.format(round(255 * config.epsilon), config.partial_rate)
            elif config.dataset == 'kadid-10k':
                savename = 'pgd_target_delta_e{}_p{}_kadid10k.pt'.format(round(255 * config.epsilon), config.partial_rate)
            else:
                savename = 'pgd_target_delta_e{}_p{}.pt'.format(round(255 * config.epsilon), config.partial_rate)
        if config.model == 'DBCNN':
            savename = savename.replace('pgd_target', 'DBCNN_pgd_target')
        torch.save(delta, savename)

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', dest='epsilon', type=float, default=0.0314, help='the scale of FGSM attacks')
    parser.add_argument('--alp', dest='alpha', type=float, default=0.00627, help='the step size of FGSM attacks')
    parser.add_argument('--step', dest='step', type=int, default=20, help='the number of steps')
    parser.add_argument('--patch_num', type=int, default=1, help='optional patch size: 10, 25')
    parser.add_argument('--target', type=float, default=50.0)

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='NT', help='path of model to attack')
    parser.add_argument('--dataset', dest='dataset', type=str, default='livec', help='Support datasets: livec') # For koniq-10k|bid|live|csiq|tid2013｜other datasets : generate corresponding data_split_file
    parser.add_argument('--dataset_path', dest='dataset_path', type=str, default='./data/ChallengeDB_release/', help='path to livec')
    parser.add_argument('--partial_rate', default=0.2, type=float)
    
    parser.add_argument('--farthest', action="store_true", default=False)
    parser.add_argument('--model', type=str, default='HyperIQA') 
    config = parser.parse_args()
    main(config)