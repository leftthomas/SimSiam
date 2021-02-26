import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Model, SimCLRLoss, MoCoLoss, NPIDLoss, DaCoLoss
from utils import DomainDataset, recall

# for reproducibility
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(net_q, data_loader, train_optimizer):
    net_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for ori_img_1, ori_img_2, gen_img_1, gen_img_2, pos_index in train_bar:
        ori_img_1, ori_img_2 = ori_img_1.cuda(gpu_ids[0]), ori_img_2.cuda(gpu_ids[0])
        _, ori_proj_1 = net_q(ori_img_1)

        if method_name == 'npid':
            loss, pos_samples = loss_criterion(ori_proj_1, pos_index)
        elif method_name == 'simclr':
            _, ori_proj_2 = net_q(ori_img_2)
            loss = loss_criterion(ori_proj_1, ori_proj_2)
        elif method_name == 'moco':
            # shuffle BN
            idx = torch.randperm(batch_size, device=ori_img_2.device)
            _, ori_proj_2 = model_k(ori_img_2[idx])
            ori_proj_2 = ori_proj_2[torch.argsort(idx)]
            loss = loss_criterion(ori_proj_1, ori_proj_2)
        else:
            # DaCo
            _, ori_proj_2 = net_q(ori_img_2)
            gen_img_1, gen_img_2 = gen_img_1.cuda(gpu_ids[0]), gen_img_2.cuda(gpu_ids[0])
            _, gen_proj_1 = net_q(gen_img_1)
            _, gen_proj_2 = net_q(gen_img_2)
            loss = loss_criterion(ori_proj_1, ori_proj_2, gen_proj_1, gen_proj_2)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        if method_name == 'npid':
            loss_criterion.enqueue(ori_proj_1, pos_index, pos_samples)
        if method_name == 'moco':
            loss_criterion.enqueue(ori_proj_2)
            # momentum update
            for parameter_q, parameter_k in zip(net_q.parameters(), model_k.parameters()):
                parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# val for one epoch
def val(net, data_loader):
    net.eval()
    vectors = []
    with torch.no_grad():
        for data, _, _, _, _ in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            vectors.append(net(data.cuda(gpu_ids[0]))[0])
        vectors = torch.cat(vectors, dim=0)
        acc_a, acc_b, acc = recall(vectors, ranks)
        precise = (acc_a[0] + acc_b[0] + acc[0]) / 3
        desc = 'Val Epoch: [{}/{}] '.format(epoch, epochs)
        for i, r in enumerate(ranks):
            results['val_ab_recall@{}'.format(r)].append(acc_a[i] * 100)
            results['val_ba_recall@{}'.format(r)].append(acc_b[i] * 100)
            results['val_cross_recall@{}'.format(r)].append(acc[i] * 100)
        desc += '| (A->B) R@{}:{:.2f}% | '.format(ranks[0], acc_a[0] * 100)
        desc += '(B->A) R@{}:{:.2f}% | '.format(ranks[0], acc_b[0] * 100)
        desc += '(A<->B) R@{}:{:.2f}% | '.format(ranks[0], acc[0] * 100)
        print(desc)
    return precise, vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='tokyo', type=str, choices=['tokyo', 'cityscapes', 'synthia'],
                        help='Dataset name')
    parser.add_argument('--method_name', default='daco', type=str, choices=['daco', 'simclr', 'moco', 'npid'],
                        help='Method name')
    parser.add_argument('--proj_dim', default=128, type=int, help='Projected feature dim for computing loss')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--iters', default=40000, type=int, help='Number of bp over the model to train')
    parser.add_argument('--gpu_ids', nargs='+', type=int, required=True, help='Selected gpus to train')
    parser.add_argument('--ranks', default='1,2,4,8', type=str, help='Selected recall')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')
    # args for DaCo
    parser.add_argument('--lamda', default=0.8, type=float, help='Lambda used for the weight of soft constrain')
    # args for NPID and MoCo
    parser.add_argument('--negs', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--momentum', default=0.5, type=float,
                        help='Momentum used for the update of memory bank or shadow model')

    # args parse
    args = parser.parse_args()
    data_root, data_name, method_name, gpu_ids = args.data_root, args.data_name, args.method_name, args.gpu_ids
    proj_dim, temperature, batch_size, iters = args.proj_dim, args.temperature, args.batch_size, args.iters
    save_root, lamda, negs, momentum = args.save_root, args.lamda, args.negs, args.momentum
    ranks = [int(k) for k in args.ranks.split(',')]

    # data prepare
    train_data = DomainDataset(data_root, data_name, split='train')
    val_data = DomainDataset(data_root, data_name, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    # compute the epochs over the dataset
    epochs = iters // (len(train_data) // batch_size)

    # model setup
    model_q = Model(proj_dim).cuda(gpu_ids[0])
    if method_name == 'moco':
        loss_criterion = MoCoLoss(negs, proj_dim, temperature).cuda(gpu_ids[0])
        model_k = Model(proj_dim).cuda(gpu_ids[0])
        # initialize model_k as a shadow model of model_q
        for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
            param_k.data.copy_(param_q.data)
            # not update by gradient
            param_k.requires_grad = False
    # optimizer config
    optimizer = Adam(model_q.parameters(), lr=1e-3, weight_decay=1e-6)
    if len(gpu_ids) > 1:
        model_q = DataParallel(model_q, device_ids=gpu_ids)
        if method_name == 'moco':
            model_k = DataParallel(model_k, device_ids=gpu_ids)

    if method_name == 'npid':
        loss_criterion = NPIDLoss(len(train_data), negs, proj_dim, momentum, temperature)
    if method_name == 'simclr':
        loss_criterion = SimCLRLoss(temperature)
    if method_name == 'daco':
        loss_criterion = DaCoLoss(lamda, temperature)

    # training loop
    results = {'train_loss': [], 'val_precise': []}
    for rank in ranks:
        results['val_ab_recall@{}'.format(rank)] = []
        results['val_ba_recall@{}'.format(rank)] = []
        results['val_cross_recall@{}'.format(rank)] = []
    save_name_pre = '{}_{}'.format(data_name, method_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model_q, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        val_precise, features = val(model_q, val_loader)
        results['val_precise'].append(val_precise * 100)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

        if val_precise > best_precise:
            best_precise = val_precise
            if len(gpu_ids) > 1:
                torch.save(model_q.module.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
            else:
                torch.save(model_q.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
            torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
