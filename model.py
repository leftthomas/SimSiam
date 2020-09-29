import torch
import torch.nn.functional as F
from pretrainedmodels import bninception
from torch import nn
from torchvision.models import resnet50, googlenet


class ProxyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProxyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # init proxy vector as unit random vector
        self.register_buffer('weight', torch.randn(out_features, in_features))

    def forward(self, x):
        normalized_weight = F.normalize(self.weight, dim=-1)
        output = x.matmul(normalized_weight.t())
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, num_classes):
        super().__init__()

        # Backbone Network
        backbones = {'resnet50': (resnet50, 2048), 'inception': (bninception, 1024), 'googlenet': (googlenet, 1024)}
        backbone, middle_dim = backbones[backbone_type]
        backbone = backbone(pretrained='imagenet' if backbone_type == 'inception' else True)
        if backbone_type == 'inception':
            backbone.global_pool = nn.AdaptiveMaxPool2d(1)
            backbone.last_linear = nn.Identity()
        else:
            backbone.avgpool = nn.AdaptiveMaxPool2d(1)
            backbone.fc = nn.Identity()
        self.backbone = backbone

        # Refactor Layer
        self.refactor = nn.Linear(middle_dim, feature_dim, bias=False)
        # Classification Layer
        self.fc = ProxyLinear(feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        feature = F.normalize(self.refactor(features), dim=-1)
        classes = self.fc(feature)
        return feature, classes

