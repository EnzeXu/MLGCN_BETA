import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch_geometric.utils import degree
import time

from dataset import MyDataset
from tqdm import tqdm

import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
# from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv


class MyNetwork(torch.nn.Module):
    def __init__(self, deg=None, in_channels=10, hidden_channels=5, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=True)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = global_add_pool(x, batch)
        return x


def train(model, args, train_loader, optimizer):
    model.train()
    # print("length: {}".format(len(train_loader)))
    total_loss = 0
    for batch_i, data in enumerate(train_loader):  # for batch_i, data in tqdm(enumerate(train_loader), total=int(len(train_loader.dataset) / args.batch_size)):
        data = data.to(args.device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        # print("data.x:", data.x.shape)
        # print("data.edge_index:", data.edge_index.shape)
        # print("data.edge_attr:", data.edge_attr.shape)
        # print("data.batch:", data.batch.shape)
        """
        data.x: torch.Size([8064, 1])
        data.edge_index: torch.Size([2, 20736])
        data.edge_attr: torch.Size([20736])
        data.batch: torch.Size([8064])
        """
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    loss = total_loss / len(train_loader.dataset)
    return model, loss


@torch.no_grad()
def test(model, args, loader):
    model.eval()
    total_error = 0
    for data in loader:
        data = data.to(args.device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    loss = total_error / len(loader.dataset)
    return loss



if __name__ == "__main__":
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'GCN_N3P')
    print(path)
    train_dataset = MyDataset(path, subset=False, split='train')
    val_dataset = MyDataset(path, subset=False, split='val')
    test_dataset = MyDataset(path, subset=False, split='test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        # print("data.num_nodes:", data.num_nodes)
        # print("data.edge_index[1]:", data.edge_index[1].shape)
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)

    t0 = time.time()
    for epoch in range(1000):
        loss = train(epoch)
        val_mae = test(val_loader)
        test_mae = test(test_loader)
        scheduler.step(val_mae)
        t = time.time()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, Test: {test_mae:.4f}, Time: {t - t0:.2f}s')
        t0 = t
