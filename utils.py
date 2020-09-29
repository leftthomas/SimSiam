import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Identity(object):
    def __call__(self, im):
        return im


class RGBToBGR(object):
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        im = Image.merge('RGB', [b, g, r])
        return im


class ScaleIntensities(object):
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __call__(self, tensor):
        tensor = (tensor - self.in_range[0]) / (self.in_range[1] - self.in_range[0]) * (
                self.out_range[1] - self.out_range[0]) + self.out_range[0]
        return tensor


class ImageReader(Dataset):

    def __init__(self, data_path, data_name, data_type, backbone_type):
        data_dict = torch.load('{}/{}/uncropped_data_dicts.pth'.format(data_path, data_name))[data_type]
        self.class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        if backbone_type == 'inception':
            normalize = transforms.Normalize([104, 117, 128], [1, 1, 1])
        else:
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if data_type == 'train':
            self.transform = transforms.Compose([
                RGBToBGR() if backbone_type == 'inception' else Identity(),
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ScaleIntensities([0, 1], [0, 255]) if backbone_type == 'inception' else Identity(),
                normalize])
        else:
            self.transform = transforms.Compose([
                RGBToBGR() if backbone_type == 'inception' else Identity(),
                transforms.Resize(292), transforms.CenterCrop(256),
                transforms.ToTensor(),
                ScaleIntensities([0, 1], [0, 255]) if backbone_type == 'inception' else Identity(),
                normalize])
        self.images, self.labels = [], []
        for label, image_list in data_dict.items():
            self.images += image_list
            self.labels += [self.class_to_idx[label]] * len(image_list)

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


def recall(feature_vectors, feature_labels, rank, gallery_vectors=None, gallery_labels=None):
    num_features = len(feature_labels)
    feature_labels = torch.tensor(feature_labels, device=feature_vectors.device)
    gallery_vectors = feature_vectors if gallery_vectors is None else gallery_vectors

    sim_matrix = torch.mm(feature_vectors, gallery_vectors.t().contiguous())

    if gallery_labels is None:
        sim_matrix.fill_diagonal_(0)
        gallery_labels = feature_labels
    else:
        gallery_labels = torch.tensor(gallery_labels, device=feature_vectors.device)

    idx = sim_matrix.topk(k=rank[-1], dim=-1, largest=True)[1]
    acc_list = []
    for r in rank:
        correct = (gallery_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / num_features).item())
    return acc_list

