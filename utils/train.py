import functools
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision

import utils.path_config as path_config


def _get_cifar_trainloader(data_dir: str,
                           dataset_fn: Callable,
                           batch_size: int) -> torch.utils.data.DataLoader:
    train_transform = transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()
                       ])
    train_dataset = dataset_fn(data_dir, train=True, download=True,
                       transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    return train_loader


def _get_cifar_testloader(data_dir: str,
                          dataset_fn: Callable,
                          batch_size: int) -> torch.utils.data.DataLoader:
    test_transform = transforms.Compose([
                           transforms.ToTensor()
                       ])
    test_dataset = dataset_fn(data_dir, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return test_loader


def _get_svhn_trainloader(data_dir: str,
                         dataset_fn: Callable,
                         batch_size: int):
    train_transform = transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()
                       ])
    split = "train"
    train_dataset = dataset_fn(data_dir,
                               split=split,
                               download=True,
                               transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    return train_loader


def _get_svhn_testloader(data_dir: str,
                         dataset_fn: Callable,
                         batch_size: int):
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    split = "test"
    test_dataset = dataset_fn(data_dir,
                              split=split,
                              download=True,
                              transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return test_loader


def get_dataloaders(dataset_name: str,
                    batch_size: int) -> Tuple[torch.utils.data.DataLoader,
                                              torch.utils.data.DataLoader]:
    paths = path_config.get_paths()
    data_dir = paths[dataset_name]

    # Choose datasets and number of classes
    if dataset_name == 'cifar100':
        dataset_fn = torchvision.datasets.CIFAR100
        train_loader_fn = functools.partial(_get_cifar_trainloader, dataset_fn=dataset_fn)
        test_loader_fn = functools.partial(_get_cifar_testloader, dataset_fn=dataset_fn)
    elif dataset_name == 'cifar10':
        dataset_fn = torchvision.datasets.CIFAR10
        train_loader_fn = functools.partial(_get_cifar_trainloader, dataset_fn=dataset_fn)
        test_loader_fn = functools.partial(_get_cifar_testloader, dataset_fn=dataset_fn)
    elif dataset_name == "svhn":
        dataset_fn = torchvision.datasets.SVHN
        train_loader_fn = functools.partial(_get_svhn_trainloader, dataset_fn=dataset_fn)
        test_loader_fn = functools.partial(_get_svhn_testloader, dataset_fn=dataset_fn)
    else:
        raise ValueError(f"Do not know about dataset = {dataset_name}")

    train_loader = train_loader_fn(data_dir=data_dir, batch_size=batch_size)
    test_loader = test_loader_fn(data_dir=data_dir, batch_size=batch_size)
    return train_loader, test_loader


def get_num_classes(dataset_name: str) -> int:
    if dataset_name in ["svhn", "cifar10"]:
        num_classes = 10
    elif dataset_name in ["cifar100"]:
        num_classes = 100
    else:
        raise ValueError(f"Do not know about dataset = {dataset_name}")
    return num_classes


def test(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         device: torch.device) -> Tuple[float, int]:
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            assert not torch.isnan(out).any()

            output = F.log_softmax(out, 1)
            total_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # print(pred)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        average_loss = total_loss / len(dataloader.dataset)
        acc = 100. * correct / len(dataloader.dataset)
    return average_loss, acc
