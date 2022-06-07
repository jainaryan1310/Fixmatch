import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import os
import torch
from torch.utils.data import Dataset
import pickle
from models.Simodel import *

from .randaugment import RandAugmentMC
from tqdm import tqdm

logger = logging.getLogger(__name__)


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
imagenet_mean = (0.4843, 0.4830, 0.4802)
imagenet_std = (0.1329, 0.1430, 0.1511)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


class Idata(Dataset):
    def __init__(self, transform=None, target_transform=None):
        super().__init__()
        filenames = [('dataset/train_data_batch_%d' % (i+1)) for i in range(1)]
        # filenames = [os.path.join('/home/ankita/scratch/datasets/imagenet32_train', f) for f in filenames]
        i = 0
        self.data = []
        self.targets = []
        for filename in filenames:
            if os.path.isfile(filename):
                res = unpickle(filename)
                if i == 0:
                    self.data = res['data'].reshape((res['data'].shape[0], 3, 32, 32))/255.
                    i += 1
                else:
                    self.data = np.concatenate((self.data, res['data'].reshape((res['data'].shape[0], 3, 32, 32))/255.), axis=0)
                self.targets += res['labels']
        self.data = self.data.transpose((0, 2, 3, 1)).astype('float32')
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray((img*255).astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return self.data.shape[0]

def dist(dz):
    "Return label distribution of selected samples"
    # create dataloader from dataset
    dl = DataLoader(dz, batch_size=1, pin_memory=True)
    d = {}

    labels = np.unique(dl.dataset.targets)
    for lbl in labels:
        d[int(lbl)] = 0
    for _, label in dl:
        d[int(label)] += 1

    return d

def get_imagenet(args, root):
    transform_labeled = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=32,
        #                       padding=int(32*0.125),
        #                       padding_mode='reflect'),
        transforms.ToTensor(),
        # transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = Idata(transform_labeled)
    base_loader = DataLoader(base_dataset, batch_size=1000, num_workers=2, shuffle=False, drop_last=False, pin_memory=False)

    victim_model = Simodel()
    state = torch.load("dataset/cifar_cnn_32.pt")
    victim_model.load_state_dict(state)
    victim_model.to(args.device)

    victim_model.eval()
    with torch.no_grad():
        for ind, (d, l) in tqdm(enumerate(base_loader)):
            d = d.to(args.device)
            l = victim_model(d).argmax(axis=1, keepdim=False)
            base_dataset.targets[ind*1000:ind*1000+int(d.shape[0])] = [int(i) for i in l.detach().cpu().tolist()]
    
    
    
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = IdataSSL(
        args, root, train_labeled_idxs, train=True, labeled=True,
        transform=transform_labeled)

    train_unlabeled_dataset = IdataSSL(
        args, root, train_unlabeled_idxs, train=True, labeled=False,
        transform=TransformFixMatch(mean=imagenet_mean, std=imagenet_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=True)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


# def get_cifar100(args, root):

#     transform_labeled = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(size=32,
#                               padding=int(32*0.125),
#                               padding_mode='reflect'),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

#     transform_val = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

#     base_dataset = datasets.CIFAR100(
#         root, train=True, download=True)

#     train_labeled_idxs, train_unlabeled_idxs = x_u_split(
#         args, base_dataset.targets)

#     train_labeled_dataset = CIFAR100SSL(
#         root, train_labeled_idxs, train=True,
#         transform=transform_labeled)

#     train_unlabeled_dataset = CIFAR100SSL(
#         root, train_unlabeled_idxs, train=True,
#         transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

#     test_dataset = datasets.CIFAR100(
#         root, train=False, transform=transform_val, download=False)

#     return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class IdataSSL():
    def __init__(self, args, root, indexs, train=True, labeled=False,
                 transform=None, target_transform=None,
                 download=False):
        
        base_dataset = Idata(transform=transform, target_transform=target_transform)
        base_loader = DataLoader(base_dataset, batch_size=1000, num_workers=2, shuffle=False, drop_last=False, pin_memory=False)
        self.labeled = labeled
        if self.labeled:
            self.model = Simodel()
            state=torch.load("dataset/cifar_cnn_32.pt")
            self.model.load_state_dict(state)
            self.model.to(args.device)
            self.model.eval()
            with torch.no_grad():
                for ind, (d, l) in tqdm(enumerate(base_loader)):
                    d = d.to(args.device)
                    l = self.model(d).argmax(axis=1, keepdim=False)
                    base_dataset.targets[ind*1000:ind*1000+int(d.shape[0])] = [int(i) for i in l.detach().cpu().tolist()]

        if indexs is not None:
            self.data = base_dataset.data[indexs]
            self.targets = np.array(base_dataset.targets)[indexs]

        self.transform = transform
        self.target_transform = target_transform



    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray((img*255).astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.targets.shape[0]


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {'imagenet': get_imagenet}

