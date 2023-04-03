# -*- encoding: utf-8 -*-
"""
@File    : train_distributed.py
@Time    : 2023/4/3 8:51
@Author  : wcirq
@Software: PyCharm
"""
import os
import time
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist

from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from model import ConvNet


def verify(network, loader, desc=""):
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # Forward pass
        outputs = network(images)
        predict = torch.argmax(outputs, dim=1)
        correct += torch.sum(labels==predict).item()
        total += images.shape[0]
    print(f"{desc} accuracy: {correct/total:0.3f} [{correct}/{total}]")


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = args.batch
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])


    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler)

    vaild_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=False,
                                               transform=transforms.ToTensor(),
                                               download=True)
    vaild_loader = torch.utils.data.DataLoader(dataset=vaild_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item())
                   )
        # vaild
        verify(model, train_loader, f"epoch{epoch}: train")
        verify(model, vaild_loader, f"epoch{epoch}: vaild")
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))

def train_nvidia_amp(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank)

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = args.batch
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    # Wrap the model
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level='O2')
    model = DDP(model)

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler)  #


    vaild_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=False,
                                               transform=transforms.ToTensor(),
                                               download=True)
    vaild_loader = torch.utils.data.DataLoader(dataset=vaild_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            ##############################################################
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            ##############################################################
            optimizer.step()

        # vaild
        verify(model, train_loader, f"epoch{epoch}: train")
        verify(model, vaild_loader, f"epoch{epoch}: vaild")
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))

def train_pytorch_amp(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank)

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = args.batch
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    model = DDP(model)

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler)  #


    vaild_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=False,
                                               transform=transforms.ToTensor(),
                                               download=True)
    vaild_loader = torch.utils.data.DataLoader(dataset=vaild_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    scaler = torch.cuda.amp.GradScaler()
    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Backward and optimize
            optimizer.zero_grad()
            # start = time.time()
            # outputs = model(images)
            # t1 = time.time() - start
            with torch.cuda.amp.autocast():
                # start = time.time()
                outputs = model(images)
                # t2 = time.time()-start
                loss = criterion(outputs, labels)
            # print(f"time1: {t1} time2: {t2}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # vaild
        verify(model, train_loader, f"epoch{epoch}: train")
        verify(model, vaild_loader, f"epoch{epoch}: vaild")
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-e', '--epochs', default=2, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch', default=2, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    mp.spawn(train_nvidia_amp, nprocs=args.gpus, args=(args,))
    mp.spawn(train_pytorch_amp, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    # python train.py -n 4 -g 8 -nr i
    main()


