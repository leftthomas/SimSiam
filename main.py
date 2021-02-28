import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Backbone, SimCLRLoss, MoCoLoss, NPIDLoss, Generator, Discriminator
from utils import DomainDataset, recall

# for reproducibility
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(net, data_loader, train_optimizer):
    net.train()
    if method_name == 'umda':
        G_content.train()
        G_style.train()
        D_content.train()
        D_style.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for ori_img_1, ori_img_2, pos_index in train_bar:
        ori_img_1, ori_img_2 = ori_img_1.cuda(gpu_ids[0]), ori_img_2.cuda(gpu_ids[0])

        if method_name == 'umda':
            # synthetic domain images
            content = G_content(ori_img_1)
            # shuffle style
            idx = torch.randperm(batch_size, device=ori_img_1.device)
            style = G_style(ori_img_1[idx])
            sytic = content + style

        _, ori_proj_1 = net(ori_img_1)

        if method_name == 'npid':
            loss, pos_samples = loss_criterion(ori_proj_1, pos_index)
        elif method_name == 'simclr':
            _, ori_proj_2 = net(ori_img_2)
            loss = loss_criterion(ori_proj_1, ori_proj_2)
        elif method_name == 'moco':
            # shuffle BN
            idx = torch.randperm(batch_size, device=ori_img_2.device)
            _, ori_proj_2 = shadow(ori_img_2[idx])
            ori_proj_2 = ori_proj_2[torch.argsort(idx)]
            loss = loss_criterion(ori_proj_1, ori_proj_2)
        else:
            # UMDA
            _, ori_proj_2 = net(sytic)
            sim_loss = loss_criterion(ori_proj_1, ori_proj_2)
            content_loss = F.mse_loss(D_content(content), D_content(sytic))
            style_loss = F.mse_loss(D_style(style), D_style(sytic))
            loss = 10 * sim_loss + content_loss + style_loss

        if method_name == 'umda':
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        if method_name == 'umda':
            optimizer_G.step()
            optimizer_D.step()

        if method_name == 'npid':
            loss_criterion.enqueue(ori_proj_1, pos_index, pos_samples)
        if method_name == 'moco':
            loss_criterion.enqueue(ori_proj_2)
            # momentum update
            for parameter_q, parameter_k in zip(net.parameters(), shadow.parameters()):
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
        for data, _, _ in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            vectors.append(net(data.cuda(gpu_ids[0]))[0])
        vectors = torch.cat(vectors, dim=0)
        acc = recall(vectors, ranks, data_loader.dataset.data_name)
        precise = acc['precise']
        desc = 'Val Epoch: [{}/{}] '.format(epoch, epochs)
        for r in ranks:
            if data_name == 'rgb':
                results['val_cf@{}'.format(r)].append(acc['cf@{}'.format(r)] * 100)
                results['val_fr@{}'.format(r)].append(acc['fr@{}'.format(r)] * 100)
                results['val_cr@{}'.format(r)].append(acc['cr@{}'.format(r)] * 100)
                results['val_cross@{}'.format(r)].append(acc['cross@{}'.format(r)] * 100)
            else:
                results['val_cd@{}'.format(r)].append(acc['cd@{}'.format(r)] * 100)
                results['val_dc@{}'.format(r)].append(acc['dc@{}'.format(r)] * 100)
                results['val_cross@{}'.format(r)].append(acc['cross@{}'.format(r)] * 100)
        if data_name == 'rgb':
            desc += '| (C<->F) R@{}:{:.2f}% | '.format(ranks[0], acc['cf@{}'.format(ranks[0])] * 100)
            desc += '(F<->R) R@{}:{:.2f}% | '.format(ranks[0], acc['fr@{}'.format(ranks[0])] * 100)
            desc += '(C<->R) R@{}:{:.2f}% | '.format(ranks[0], acc['cr@{}'.format(ranks[0])] * 100)
            desc += '(Cross) R@{}:{:.2f}% | '.format(ranks[0], acc['cross@{}'.format(ranks[0])] * 100)
        else:
            desc += '| (C->D) R@{}:{:.2f}% | '.format(ranks[0], acc['cd@{}'.format(ranks[0])] * 100)
            desc += '(D->C) R@{}:{:.2f}% | '.format(ranks[0], acc['dc@{}'.format(ranks[0])] * 100)
            desc += '(C<->D) R@{}:{:.2f}% | '.format(ranks[0], acc['cross@{}'.format(ranks[0])] * 100)
        print(desc)
    return precise, vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='rgb', type=str, choices=['rgb', 'modal'], help='Dataset name')
    parser.add_argument('--method_name', default='umda', type=str, choices=['umda', 'simclr', 'moco', 'npid'],
                        help='Method name')
    parser.add_argument('--proj_dim', default=128, type=int, help='Projected feature dim for computing loss')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--iters', default=40000, type=int, help='Number of bp over the model to train')
    parser.add_argument('--gpu_ids', nargs='+', type=int, required=True, help='Selected gpus to train')
    parser.add_argument('--ranks', default='1,2,4,8', type=str, help='Selected recall')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')
    # args for NPID and MoCo
    parser.add_argument('--negs', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--momentum', default=0.5, type=float,
                        help='Momentum used for the update of memory bank or shadow model')

    # args parse
    args = parser.parse_args()
    data_root, data_name, method_name, gpu_ids = args.data_root, args.data_name, args.method_name, args.gpu_ids
    proj_dim, temperature, batch_size, iters = args.proj_dim, args.temperature, args.batch_size, args.iters
    save_root, negs, momentum = args.save_root, args.negs, args.momentum
    ranks = [int(k) for k in args.ranks.split(',')]

    # data prepare
    train_data = DomainDataset(data_root, data_name, split='train')
    val_data = DomainDataset(data_root, data_name, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    # compute the epochs over the dataset
    epochs = iters // (len(train_data) // batch_size)

    # model setup
    backbone = Backbone(proj_dim).cuda(gpu_ids[0])
    if method_name == 'umda':
        G_content = Generator(3, 3).cuda(gpu_ids[0])
        G_style = Generator(3, 3).cuda(gpu_ids[0])
        D_content = Discriminator(3).cuda(gpu_ids[0])
        D_style = Discriminator(3).cuda(gpu_ids[0])
    if method_name == 'moco':
        loss_criterion = MoCoLoss(negs, proj_dim, temperature).cuda(gpu_ids[0])
        shadow = Backbone(proj_dim).cuda(gpu_ids[0])
        # initialize shadow as a shadow model of backbone
        for param_q, param_k in zip(backbone.parameters(), shadow.parameters()):
            param_k.data.copy_(param_q.data)
            # not update by gradient
            param_k.requires_grad = False
    # optimizer config
    optimizer_backbone = Adam(backbone.parameters(), lr=1e-3, weight_decay=1e-6)
    if method_name == 'umda':
        optimizer_G = Adam(itertools.chain(G_content.parameters(), G_style.parameters()), lr=1e-3, betas=(0.5, 0.999))
        optimizer_D = Adam(itertools.chain(D_content.parameters(), D_style.parameters()), lr=1e-4, betas=(0.5, 0.999))
    if len(gpu_ids) > 1:
        backbone = DataParallel(backbone, device_ids=gpu_ids)
        if method_name == 'moco':
            shadow = DataParallel(shadow, device_ids=gpu_ids)
        if method_name == 'umda':
            G_content = DataParallel(G_content, device_ids=gpu_ids)
            G_style = DataParallel(G_style, device_ids=gpu_ids)
            D_content = DataParallel(D_content, device_ids=gpu_ids)
            D_style = DataParallel(D_style, device_ids=gpu_ids)

    if method_name == 'npid':
        loss_criterion = NPIDLoss(len(train_data), negs, proj_dim, momentum, temperature)
    if method_name in ['simclr', 'umda']:
        loss_criterion = SimCLRLoss(temperature)

    # training loop
    results = {'train_loss': [], 'val_precise': []}
    for rank in ranks:
        if data_name == 'rgb':
            results['val_cf@{}'.format(rank)] = []
            results['val_fr@{}'.format(rank)] = []
            results['val_cr@{}'.format(rank)] = []
            results['val_cross@{}'.format(rank)] = []
        else:
            results['val_cd@{}'.format(rank)] = []
            results['val_dc@{}'.format(rank)] = []
            results['val_cross@{}'.format(rank)] = []
    save_name_pre = '{}_{}'.format(data_name, method_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(backbone, train_loader, optimizer_backbone)
        results['train_loss'].append(train_loss)
        val_precise, features = val(backbone, val_loader)
        results['val_precise'].append(val_precise * 100)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

        if val_precise > best_precise:
            best_precise = val_precise
            if len(gpu_ids) > 1:
                torch.save(backbone.module.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
            else:
                torch.save(backbone.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
            torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
