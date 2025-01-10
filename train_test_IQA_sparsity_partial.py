import os
import argparse
import random
import numpy as np
import sys
sys.path.append('.')
from HyperIQASolver_benign import HyperIQASolver
from DBCNNSolver_benign import DBCNNSolver
import torch
import pickle

import sys
class Logger(object):
    def __init__(self, fileN="livec_bs16.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        pass


def fix_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(config):

    folder_path = {config.dataset:config.dataset_path}

    expid = config.expid
    print('expid', expid)
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
    srcc_all = np.zeros(config.train_test_num, dtype=float)
    plcc_all = np.zeros(config.train_test_num, dtype=float)

    sys.stdout = Logger('livec_bs16.log')
    print('Dataset: %s, Batch Size: %i, If Grad: %s, Weight: %e.' % (config.dataset, config.batch_size, config.if_grad, config.weight))
    for i in range(config.train_test_num):
        print('Round %d' % (i+1))
        if config.dataset == 'livec':
            train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
            test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        if config.exist_partial_index:
            if config.dataset == 'koniq-10k':
                selected_index = np.load('partial_index_p{}_koniq10k.npy'.format(config.partial_rate))
            elif config.dataset == 'kadid-10k':
                selected_index = np.load('partial_index_p{}_kadid10k.npy'.format(config.partial_rate))
            else:
                selected_index = np.load('partial_index_p{}.npy'.format(config.partial_rate))
        else:
            selected_index = np.random.choice(len(train_index), int(len(train_index)*config.partial_rate), replace=False)
            if config.dataset == 'koniq-10k':
                np.save('partial_index_p{}_koniq10k.npy'.format(config.partial_rate), selected_index)
            elif config.dataset == 'kadid-10k':
                np.save('partial_index_p{}_kadid10k.npy'.format(config.partial_rate), selected_index)
            else:
                np.save('partial_index_p{}.npy'.format(config.partial_rate), selected_index)
        if config.model == 'HyperIQA':
            solver = HyperIQASolver(config, folder_path[config.dataset], train_index, test_index, partial = True)
        elif config.model == 'DBCNN':
            solver = DBCNNSolver(config, folder_path[config.dataset], train_index, test_index, partial = True)
        srcc_all[i], plcc_all[i] = solver.train()

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='livec', help='Support datasets: livec') # For koniq-10k|bid|live|csiq|tid2013｜other datasets : generate corresponding data_split_file
    parser.add_argument('--dataset_path', dest='dataset_path', type=str, default='./data/ChallengeDB_release/', help='path to livec')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=24, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')
    parser.add_argument('--expid', dest='expid', type=int, default=0, help='Exp ID')

    parser.add_argument('--if_grad', action="store_true", help='whether use l1 loss')
    parser.add_argument('--weight', type=float, default=1e-3, help='The weight of gradient loss')
    parser.add_argument('--h', type=float, default=1e-2, help='the step size when approximating the gradient')

    parser.add_argument('--partial_rate', default=0.2, type=float)
    parser.add_argument('--exist_partial_index', action="store_true", default = False)
    parser.add_argument('--model', type=str, default='HyperIQA') 
    config = parser.parse_args()
    main(config)

