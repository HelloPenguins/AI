import os
import random
import glob
import re

from importlib import import_module
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torchvision import models

import dataset


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = os.path.join(model_dir, args.name)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- augmentation
    augmentation_module = getattr(import_module("dataset"), args.augmentation)
    transform = augmentation_module(
        resize=args.resize,
        mean=1,
        std=0.5,
    )

    # -- dataset
    PillBaseDataset = dataset.PillBaseDataset(data_dir)
    train_set, valid_set = PillBaseDataset.getDataset(transform.transform)
    train_loader, val_loader  = PillBaseDataset.getDataloader(train_set, valid_set, args.batch_size)

    num_classes = len(train_set.classes)

    model = build_model(num_classes, args)

    # -- loss & metric
    criterion = nn.CrossEntropyLoss()
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    
    for epoch in range(args.epochs):
        model.train()

        batch_length = len(train_loader)
        avg_loss = 0
        avg_acc = 0
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / batch_length
            avg_acc += (preds == labels).float().mean() / batch_length

        print('epoch: {}, loss: {:.6f}, acc: {:.6f}'.format(epoch+1, loss.item(), avg_acc))

        model.eval()
        with torch.no_grad():
            batch_length = len(val_loader)
            val_avg_score = 0
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_avg_score += (preds == labels).float().mean() / batch_length
            print("test accuracy: {:.4f}".format(val_avg_score.item()))
        torch.save(model.state_dict(), f"{save_dir}/{epoch}.pth")

def build_model(num_classes, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- model  
    if args.model == "vgg16":
        model = models.vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False

        model.classifier[-1] = nn.Sequential(
            nn.Linear(model.classifier[-1].in_features, num_classes),
        )

        model.to(device)
    return model

