'''
Implementation of Barlow Twins (https://arxiv.org/abs/2103.03230) adaptation
'''

from metrics import Metrics

import random
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from warnings import filterwarnings
from IPython.display import clear_output
from tqdm import tqdm

filterwarnings('ignore')


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsAdaptation(nn.Module):
    '''
    Adapted from https://github.com/facebookresearch/barlowtwins
    '''
    def __init__(self, backbone, projection_sizes, lambd):
        '''

        :param backbone: Model backbone
        :param projection_sizes: size of the hidden layers in the projection MLP
        :param lambd: tradeoff function
        '''
        super().__init__()
        self.lambd = lambd

        # backbone (ResNet50)
        self.backbone = backbone
        self.backbone.fc = nn.Identity()

        # projector
        sizes = projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        if len(sizes) >= 2:
          layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        # pairwise similarity net
        self.sim = nn.Sequential(
            nn.Linear(sizes[-1] * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        return self.sim(torch.cat((z1, z2), dim=1)).squeeze(1)

    def forward_all_pairs(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        z1_repeated = z1.repeat_interleave(z2.shape[0], dim=0)
        z2_tiled = z2.repeat(z1.shape[0], 1)

        all_pairs = torch.cat((z1_repeated, z2_tiled), dim=1)
        return self.sim(all_pairs).squeeze(1)

    def forward_projector(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = torch.mm(self.bn(z1).T, self.bn(z2))
        c.div_(z1.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambd * off_diag

    def adjust_learning_rate(self, optimizer, start_lr, end_lr, epoch, epochs):
        coef = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
        lr = start_lr * coef + end_lr * (1 - coef)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train_projector(self, train_dataloader, validation_dataloader,
                   optimizer, start_lr=1e-3, end_lr=1e-3, epochs=10, plot=True):
        '''
        Training projector with frozen backbone's weights and similarity net

        train_dataloader: dataloader for train samples
        validation_dataloader: dataloader for validation samples
        start_lr: start learning rate (for lr adjustment)
        end_lr: end learning rate (for lr adjustment)
        optimizer: gradient descent optimizer
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        print("Training on", device)
        self.backbone.requires_grad_(False)
        self.projector.requires_grad_(True)
        self.bn.requires_grad_(True)
        self.sim.requires_grad_(False)
        train_losses = []
        validation_losses = []
        for epoch in tqdm(range(epochs)):
            total_train_loss = 0
            for batch_idx, x in enumerate(train_dataloader):
                x = x.to(device)
                x1 = x[:, 0, :, :, :]
                x2 = x[:, 1, :, :, :]
                loss = self.forward_projector(x1, x2)
                total_train_loss = (total_train_loss * batch_idx + loss.item()) / (batch_idx + 1)
                self.adjust_learning_rate(optimizer, start_lr, end_lr, epoch, epochs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_validation_loss = 0
            for batch_idx, x in enumerate(validation_dataloader):
                x = x.to(device)
                x1 = x[:, 0, :, :, :]
                x2 = x[:, 1, :, :, :]
                loss = self.forward_projector(x1, x2)
                total_validation_loss = (total_validation_loss * batch_idx + loss.item()) / (batch_idx + 1)

            if epoch == 0:
                continue

            train_losses.append(total_train_loss)
            validation_losses.append(total_validation_loss)

            if plot:
                clear_output()
                plt.figure(figsize=(10, 6))
                plt.title("Training Projector")
                plt.plot(train_losses, color='blue', label="train")
                plt.plot(validation_losses, color='orange', label="validation")
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.legend()
                plt.show()

        return train_losses, validation_losses

    def train_similarity_net(self, train_dataloader, validation_dataloader,
                   optimizer, start_lr=1e-3, end_lr=1e-3, epochs=10, plot=True):
        '''
        Training similarity net with whole model frozen

        train_dataloader: dataloader for train samples
        validation_dataloader: dataloader for validation samples
        start_lr: start learning rate (for lr adjustment)
        end_lr: end learning rate (for lr adjustment)
        optimizer: gradient descent optimizer
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        print("Training on", device)
        self.backbone.requires_grad_(False)
        self.projector.requires_grad_(False)
        self.bn.requires_grad_(False)
        self.sim.requires_grad_(True)
        criterion = nn.BCELoss()
        train_losses = []
        train_acc = []
        validation_losses = []
        validation_acc = []
        for epoch in tqdm(range(epochs)):
            total_train_loss = 0
            mean_train_acc = 0
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x = x.to(device)
                target = target.to(device)
                x1 = x[:, 0, :, :, :]
                x2 = x[:, 1, :, :, :]
                prob = self(x1, x2)
                loss = criterion(prob, target)
                total_train_loss = (total_train_loss * batch_idx + loss.item()) / (batch_idx + 1)
                mean_train_acc = (mean_train_acc * batch_idx + Metrics.f1_score(prob, target)) / (batch_idx + 1)
                self.adjust_learning_rate(optimizer, start_lr, end_lr, epoch, epochs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_validation_loss = 0
            mean_validation_acc = 0
            for batch_idx, (x, target) in enumerate(validation_dataloader):
                x = x.to(device)
                target = target.to(device)
                x1 = x[:, 0, :, :, :]
                x2 = x[:, 1, :, :, :]
                prob = self(x1, x2)
                loss = criterion(prob, target)
                total_validation_loss = (total_validation_loss * batch_idx + loss.item()) / (batch_idx + 1)
                mean_validation_acc = (mean_validation_acc * batch_idx + Metrics.f1_score(prob, target)) / (batch_idx + 1)

            if epoch == 0:
                continue

            train_losses.append(total_train_loss)
            validation_losses.append(total_validation_loss)
            train_acc.append(mean_train_acc)
            validation_acc.append(mean_validation_acc)

            if plot:
                clear_output()
                plt.figure(figsize=(14, 6))
                plt.suptitle("Training Similarity Net")

                plt.subplot(1, 2, 1)
                plt.plot(train_losses, color='blue', label="train")
                plt.plot(validation_losses, color='orange', label="validation")
                plt.xlabel('Epoch')
                plt.ylabel('BCE Loss')
                plt.grid(True)
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(train_acc, color='blue', label="train")
                plt.plot(validation_acc, color='orange', label="validation")
                plt.xlabel('Epoch')
                plt.ylabel('F1 Score')
                plt.grid(True)
                plt.legend()
                plt.show()

        return train_losses, validation_losses, train_acc, validation_acc

    def train_projector_and_similarity_net_with_combined_loss(self, train_dataloader, validation_dataloader,
                   optimizer, start_lr=1e-3, end_lr=1e-3, lambd=1e-3, epochs=10, plot=True):
        '''
        Training projector and similarity net with combined loss

        train_dataloader: dataloader for train samples
        validation_dataloader: dataloader for validation samples
        optimizer: gradient descent optimizer
        start_lr: start learning rate (for lr adjustment)
        lambd: tradeoff factor for the combined loss
        end_lr: end learning rate (for lr adjustment)
        '''

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        print("Training on", device)
        self.backbone.requires_grad_(False)
        self.projector.requires_grad_(True)
        self.bn.requires_grad_(True)
        self.sim.requires_grad_(True)

        def criterion(prob, target, x1, x2):
            return nn.BCELoss()(prob, target) + lambd * self.forward_projector(x1, x2)

        train_losses = []
        train_acc = []
        validation_losses = []
        validation_acc = []

        for epoch in tqdm(range(epochs)):
            total_train_loss = 0
            mean_train_acc = 0
            for batch_idx, x in enumerate(train_dataloader):
                x = x.to(device)
                x1 = x[:, 0, :, :, :]
                x2 = x[:, 1, :, :, :]

                prob = self.forward_all_pairs(x1, x2)
                target = torch.eye(x1.shape[0]).flatten().to(device)

                loss = criterion(prob, target, x1, x2)
                total_train_loss = (total_train_loss * batch_idx + loss.item()) / (batch_idx + 1)
                mean_train_acc = (mean_train_acc * batch_idx + Metrics.f1_score(prob, target)) / (batch_idx + 1)

                self.adjust_learning_rate(optimizer, start_lr, end_lr, epoch, epochs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_validation_loss = 0
            mean_validation_acc = 0
            for batch_idx, x in enumerate(validation_dataloader):
                x = x.to(device)
                x1 = x[:, 0, :, :, :]
                x2 = x[:, 1, :, :, :]

                prob = self.forward_all_pairs(x1, x2)
                target = torch.eye(x1.shape[0]).flatten().to(device)

                loss = criterion(prob, target, x1, x2)
                total_validation_loss = (total_validation_loss * batch_idx + loss.item()) / (batch_idx + 1)
                mean_validation_acc = (mean_validation_acc * batch_idx + Metrics.f1_score(prob, target)) / (batch_idx + 1)

            if epoch == 0:
                continue

            train_losses.append(total_train_loss)
            validation_losses.append(total_validation_loss)
            train_acc.append(mean_train_acc)
            validation_acc.append(mean_validation_acc)

            if plot:
                clear_output()
                plt.figure(figsize=(14, 6))
                plt.suptitle(f"Training Projector and Similarity Net with Combined Loss (lambd = {lambd})")

                plt.subplot(1, 2, 1)
                plt.plot(train_losses, color='blue', label="train")
                plt.plot(validation_losses, color='orange', label="validation")
                plt.xlabel('Epoch')
                plt.ylabel('Combined Loss')
                plt.grid(True)
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(train_acc, color='blue', label="train")
                plt.plot(validation_acc, color='orange', label="validation")
                plt.xlabel('Epoch')
                plt.ylabel('F1 Score')
                plt.grid(True)
                plt.legend()
                plt.show()

        return train_losses, validation_losses
