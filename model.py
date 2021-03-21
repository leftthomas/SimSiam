import torch
import torch.nn as nn
from torchvision.models import resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=2048):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18(zero_init_residual=True).named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 2048, bias=False), nn.BatchNorm1d(2048),
                               nn.ReLU(inplace=True), nn.Linear(2048, feature_dim, bias=False),
                               nn.BatchNorm1d(feature_dim))
        # prediction head
        self.h = nn.Sequential(nn.Linear(feature_dim, feature_dim // 4, bias=False), nn.BatchNorm1d(feature_dim // 4),
                               nn.ReLU(inplace=True), nn.Linear(feature_dim // 4, feature_dim, bias=True))

    def forward(self, x):
        x = torch.flatten(self.f(x), start_dim=1)
        feature = self.g(x)
        proj = self.h(feature)
        return feature, proj
