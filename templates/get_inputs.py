import torch
from torch import nn
from torchvision import datasets, transforms

from Tools.cli import flags
from Tools.utils_torch import infinity_loader

FLAGS = flags.FLAGS


def get_optimizer(params, opt_name, lr, beta1, beta2, weight_decay=0.0):
    if opt_name.lower() == "adam":
        optim = torch.optim.Adam(params, lr, betas=(beta1, beta2))
    elif opt_name.lower() == "nesterov":
        optim = torch.optim.SGD(
            params, lr, momentum=beta1, weight_decay=weight_decay, nesterov=True
        )
    return optim


def get_dataset(train):
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    if FLAGS.dataset.lower() == "svhn":
        if train is True:
            split = "train"
        else:
            split = "test"

        sets = datasets.SVHN(
            "/home/LargeData/svhn", split=split, download=True, transform=transf
        )
    elif FLAGS.dataset.lower() == "cifar10":
        sets = datasets.CIFAR10(
            "/home/LargeData/cifar", train=train, download=True, transform=transf
        )
    return sets


def get_dataloader(batch_size, dataset=None, train=True, infinity=True):
    if dataset is None:
        dataset = get_dataset(train)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size, drop_last=True, shuffle=True, num_workers=8,
    )
    if infinity is True:
        return infinity_loader(loader)
    else:
        return loader
