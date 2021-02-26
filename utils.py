import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

normalizer = {'rgb': [(0.401, 0.437, 0.401), (0.186, 0.187, 0.187)],
              'modal': [(0.244, 0.221, 0.177), (0.193, 0.174, 0.143)]}


def get_transform(data_name, split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(normalizer[data_name][0], normalizer[data_name][1])])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(normalizer[data_name][0], normalizer[data_name][1])])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, split='train'):
        super(DomainDataset, self).__init__()

        self.data_name = data_name
        image_path = os.path.join(data_root, data_name, split, '*', '*', '*.png')
        self.images = glob.glob(image_path)
        self.images.sort()
        self.transform = get_transform(data_name, split)

    def __getitem__(self, index):
        img_name = self.images[index]
        img = Image.open(img_name)
        img_1 = self.transform(img)
        img_2 = self.transform(img)
        return img_1, img_2, index

    def __len__(self):
        return len(self.images)


def recall(vectors, ranks, data_name):
    labels = torch.arange(len(vectors) // 2, device=vectors.device)
    labels = torch.cat((labels, labels), dim=0)
    a_vectors = vectors[:len(vectors) // 2]
    b_vectors = vectors[len(vectors) // 2:]
    a_labels = labels[:len(vectors) // 2]
    b_labels = labels[len(vectors) // 2:]
    # domain a ---> domain b
    sim_a = a_vectors.mm(b_vectors.t())
    idx_a = sim_a.topk(k=ranks[-1], dim=-1, largest=True)[1]
    # domain b ---> domain a
    sim_b = b_vectors.mm(a_vectors.t())
    idx_b = sim_b.topk(k=ranks[-1], dim=-1, largest=True)[1]
    # cross domain
    sim = vectors.mm(vectors.t())
    sim.fill_diagonal_(-np.inf)
    idx = sim.topk(k=ranks[-1], dim=-1, largest=True)[1]

    acc_a, acc_b, acc = [], [], []
    for r in ranks:
        correct_a = (torch.eq(b_labels[idx_a[:, 0:r]], a_labels.unsqueeze(dim=-1))).any(dim=-1)
        acc_a.append((torch.sum(correct_a) / correct_a.size(0)).item())
        correct_b = (torch.eq(a_labels[idx_b[:, 0:r]], b_labels.unsqueeze(dim=-1))).any(dim=-1)
        acc_b.append((torch.sum(correct_b) / correct_b.size(0)).item())
        correct = (torch.eq(labels[idx[:, 0:r]], labels.unsqueeze(dim=-1))).any(dim=-1)
        acc.append((torch.sum(correct) / correct.size(0)).item())
    return acc_a, acc_b, acc
