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
from network import Add_trigger_net_v2, Add_trigger_net_v3, Add_trigger_net_v4
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
FGSM--untargeted attack (UAP)
model: the NR-IQA model
x: input image
pred_score: predicted socre of the unattaked image
eps: the l_inf norm bound of the perturbation
alpha: the step size in the I-FGSM attack
iteration: the iteration number of the I-FGSM attack
'''
def IQA_untarget_UAP_DCT(optimizer, add_trigger, t, model, x, alpha=10, iteration=10, epsilon = 0.01568):
    with torch.no_grad():
        pred_score = model(norm(x)).detach()
    t_adv = Variable(t.data, requires_grad=True)
    if optimizer == 'ifgsm':
        for i in range(iteration):
            x_adv = add_trigger(x, t_adv)
            score_adv = model(norm(x_adv))
            loss_pred = nn.MSELoss()(score_adv, pred_score) # L_{mse}
            loss_visibility = max(nn.MSELoss()(x_adv, x), epsilon**2)
            loss = - loss_pred + 10000**2 * loss_visibility
            model.zero_grad()
            if t_adv.grad is not None:
                t_adv.grad.data.fill_(0)
            loss.backward(retain_graph=True)

            t_adv.grad.sign_()
            t_adv = t_adv - alpha*t_adv.grad
            t_adv = Variable(t_adv.data, requires_grad=True)
    else:
        if optimizer == 'adam':
            opt = optim.Adam([t_adv], lr=1)
        elif optimizer == 'sgd':
            opt = optim.SGD([t_adv], lr=10, momentum=0.9)
        for i in range(iteration):
            def closure():
                opt.zero_grad()
                x_adv = add_trigger(x, t_adv)
                score_adv = model(norm(x_adv))
                loss_pred = nn.MSELoss()(score_adv, pred_score) # L_{mse}
                loss_visibility = max(nn.MSELoss()(x_adv, x), epsilon**2)
                loss = - loss_pred + 10000**2 * loss_visibility
                loss.backward()
                return loss
            opt.step(closure)

    score_org = pred_score
    with torch.no_grad():
        x_adv = add_trigger(x, t_adv)
        score_adv = model(norm(x_adv))
        loss_visibility = nn.MSELoss()(x_adv, x)

    return t_adv.detach(), score_adv, score_org, loss_visibility


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
    train_loader = data_loader.DataLoader(config.dataset, folder_path[config.dataset], train_index, 224, config.patch_num, batch_size=128, istrain=True, poison = True, poison_rate = 0.0)

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

    train_data = train_loader.get_data()

    if config.mode == 'mid':
        add_trigger = Add_trigger_net_v2(trigger_channel = config.channels)
        t = torch.zeros(3, config.channels).uniform_(-300, 300).cuda()
    elif config.mode == 'high':
        add_trigger = Add_trigger_net_v3(trigger_channel = config.channels)
        t = torch.zeros(3, config.channels).uniform_(-300, 300).cuda()
    elif config.mode == 'mix':
        add_trigger = Add_trigger_net_v4(trigger_channel = config.channels)
        t = torch.zeros(3, config.channels//2).uniform_(-300, 300).cuda()


    for e in range(config.epoch):
        score_i = []
        score_ori_i = []
        for img, label, mark in train_data:
            img = poison_transform(img).cuda()
            t, pert_score, org_score, loss_visibility = IQA_untarget_UAP_DCT(config.optimizer, add_trigger, t, model, img, alpha=config.alpha, iteration=config.step, epsilon = config.epsilon / 255.0)
            score_i.append(pert_score.detach().cpu().numpy().mean().item())
            score_ori_i.append(org_score.detach().cpu().numpy().mean().item())
            print ([org_score.mean(), pert_score.mean(), loss_visibility])

        score_i = np.array(score_i)
        score_i = np.mean(score_i)
        score_ori_i = np.array(score_ori_i)
        score_ori_i = np.mean(score_ori_i)
            
        print('Predicted Score (Unattacked):{}'.format(score_ori_i))
        print('Predicted Score (Attacked):{}'.format(score_i))
        if config.farthest:
            savename = 'trigger_p{}_{}_farthest.pt'.format(config.partial_rate, config.optimizer)
        else:
            if config.dataset == 'koniq-10k':
                savename = 'trigger_p{}_{}_koniq10k.pt'.format(config.partial_rate, config.optimizer)
            elif config.dataset == 'kadid-10k':
                savename = 'trigger_p{}_{}_kadid10k.pt'.format(config.partial_rate, config.optimizer)
            else:
                savename = 'trigger_p{}_{}.pt'.format(config.partial_rate, config.optimizer)
        if config.model == 'DBCNN':
            savename = savename.replace('trigger', 'DBCNN_trigger')
        if config.epsilon > 4.0:
            savename = savename.replace('.pt', '_e{}.pt'.format(int(config.epsilon)))
        if config.mode in ['high', 'mix']:
            savename = savename.replace('.pt', '_{}.pt'.format(config.mode))
        if config.channels != 64:
            savename = savename.replace('.pt', '_c{}.pt'.format(config.channels))
        torch.save(t.cpu().detach(), savename)
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alp', dest='alpha', type=float, default=1, help='the step size of FGSM attacks')
    parser.add_argument('--step', dest='step', type=int, default=10, help='the number of steps')
    parser.add_argument('--patch_num', type=int, default=25, help='optional patch size: 10, 25')

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--model_path', type=str, default='NT', help='path of model to attack')
    parser.add_argument('--dataset', dest='dataset', type=str, default='livec', help='Support datasets: livec') # For koniq-10k|bid|live|csiq|tid2013｜other datasets : generate corresponding data_split_file
    parser.add_argument('--dataset_path', dest='dataset_path', type=str, default='./data/ChallengeDB_release/', help='path to livec')
    parser.add_argument('--partial_rate', default=0.2, type=float)
    parser.add_argument('--optimizer', default='ifgsm', type=str)
    parser.add_argument('--epsilon', default=4.0, type=float)
    
    parser.add_argument('--farthest', action="store_true", default=False)
    parser.add_argument('--model', type=str, default='HyperIQA') 
    parser.add_argument('--mode', type=str, default='mid', choices=['mid', 'high', 'mix']) 
    parser.add_argument('--channels', type=int, default=64)
    config = parser.parse_args()
    main(config)