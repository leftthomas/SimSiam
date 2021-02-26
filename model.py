import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, proj_dim):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, proj_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        proj = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(proj, dim=-1)


class SimCLRLoss(nn.Module):
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, proj_1, proj_2):
        batch_size = proj_1.size(0)
        # [2*B, Dim]
        out = torch.cat([proj_1, proj_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(proj_1 * proj_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


class MoCoLoss(nn.Module):
    def __init__(self, negs, proj_dim, temperature):
        super(MoCoLoss, self).__init__()
        # init memory queue as unit random vector ---> [Negs, Dim]
        self.register_buffer('queue', F.normalize(torch.randn(negs, proj_dim), dim=-1))
        self.temperature = temperature

    def forward(self, query, key):
        batch_size = query.size(0)
        # [B, 1]
        score_pos = torch.sum(query * key, dim=-1, keepdim=True)
        # [B, Negs]
        score_neg = torch.mm(query, self.queue.t().contiguous())
        # [B, 1+Negs]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / self.temperature, torch.zeros(batch_size, dtype=torch.long, device=query.device))
        return loss

    def enqueue(self, key):
        # update queue
        self.queue.copy_(torch.cat((self.queue, key), dim=0)[key.size(0):])


class NPIDLoss(nn.Module):
    def __init__(self, n, negs, proj_dim, momentum, temperature):
        super(NPIDLoss, self).__init__()
        self.n = n
        self.negs = negs
        self.proj_dim = proj_dim
        self.momentum = momentum
        self.temperature = temperature
        # init memory bank as unit random vector ---> [N, Dim]
        self.register_buffer('bank', F.normalize(torch.randn(n, proj_dim), dim=-1))
        # z as normalizer, init with None
        self.z = None

    def forward(self, proj, pos_index):
        batch_size = proj.size(0)
        # randomly generate Negs+1 sample indexes for each batch ---> [B, Negs+1]
        idx = torch.randint(high=self.n, size=(batch_size, self.negs + 1))
        # make the first sample as positive
        idx[:, 0] = pos_index
        # select memory vectors from memory bank ---> [B, 1+Negs, Dim]
        samples = torch.index_select(self.bank, dim=0, index=idx.view(-1)).view(batch_size, -1, self.proj_dim)
        # compute cos similarity between each feature vector and memory bank ---> [B, 1+Negs]
        sim_matrix = torch.bmm(samples.to(device=proj.device), proj.unsqueeze(dim=-1)).view(batch_size, -1)
        out = torch.exp(sim_matrix / self.temperature)
        # Monte Carlo approximation, use the approximation derived from initial batches as z
        if self.z is None:
            self.z = out.detach().mean().item() * self.n
        # compute P(i|v) ---> [B, 1+Negs]
        output = out / self.z

        # compute loss
        # compute log(h(i|v))=log(P(i|v)/(P(i|v)+Negs*P_n(i))) ---> [B]
        p_d = (output.select(dim=-1, index=0) / (output.select(dim=-1, index=0) + self.negs / self.n)).log()
        # compute log(1-h(i|v'))=log(1-P(i|v')/(P(i|v')+Negs*P_n(i))) ---> [B, Negs]
        p_n = ((self.negs / self.n) / (output.narrow(dim=-1, start=1, length=self.negs) + self.negs / self.n)).log()
        # compute J_NCE(θ)=-E(P_d)-Negs*E(P_n)
        loss = - (p_d.sum() + p_n.sum()) / batch_size

        pos_samples = samples.select(dim=1, index=0)
        return loss, pos_samples

    def enqueue(self, proj, pos_index, pos_samples):
        # update memory bank ---> [B, Dim]
        pos_samples = proj.detach().cpu() * self.momentum + pos_samples * (1.0 - self.momentum)
        pos_samples = F.normalize(pos_samples, dim=-1)
        self.bank.index_copy_(0, pos_index, pos_samples)


class DaCoLoss(nn.Module):
    def __init__(self, lamda, temperature):
        super(DaCoLoss, self).__init__()
        self.lamda = lamda
        self.temperature = temperature
        self.base_loss = SimCLRLoss(temperature)

    def forward(self, ori_proj_1, ori_proj_2, gen_proj_1, gen_proj_2):
        within_domain_loss = self.base_loss(ori_proj_1, ori_proj_2) + self.base_loss(gen_proj_1, gen_proj_2)
        cross_domain_loss = self.base_loss(ori_proj_1, gen_proj_1) + self.base_loss(ori_proj_1, gen_proj_2)
        loss = within_domain_loss + self.lamda * cross_domain_loss
        return loss
