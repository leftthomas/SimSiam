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
        normalized_x = F.normalize(x, dim=-1)
        normalized_weight = F.normalize(self.weight, dim=-1)
        output = normalized_x.mm(normalized_weight.t())
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, k, num_classes):
        super().__init__()

        assert feature_dim % k == 0, 'feature_dim must be divided by k'

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
        self.k = k
        self.refactor = nn.ModuleList([nn.Linear(middle_dim, feature_dim // k, bias=False) for _ in range(k)])

        # Classification Layer
        self.fc = nn.ModuleList([ProxyLinear(feature_dim // k, num_classes) for _ in range(k)])

    def forward(self, x):
        features = self.backbone(x)
        global_feature = F.layer_norm(features, features.size()[1:])
        embeddings, norms, outputs = [], [], []
        for i in range(self.k):
            # [B, D/K]
            feature = self.refactor[i](global_feature)
            embeddings.append(feature)
            norm = torch.norm(feature, dim=-1, keepdim=True)
            norms.append(norm)
            classes = self.fc[i](feature)
            outputs.append(classes)
        # [B, K, D/K], [B, K, 1]
        embeddings, norms = torch.stack(embeddings, dim=1), torch.stack(norms, dim=1)
        # [B, K, N]
        outputs = torch.stack(outputs, dim=1) * (norms / torch.sum(norms, dim=1, keepdim=True))
        return embeddings, outputs
