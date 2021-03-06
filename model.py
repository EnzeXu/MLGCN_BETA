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


class MyNetwork(torch.nn.Module):
    def __init__(self, _deg):
        super().__init__()

        # self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 10)  # self.edge_emb = Embedding(4, 50)

        self.conv_num = 2
        self.in_num = 10

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        # for _ in range(4):
        #     conv = PNAConv(in_channels=75, out_channels=75,
        #                    aggregators=aggregators, scalers=scalers, deg=_deg,
        #                    edge_dim=50, towers=5, pre_layers=1, post_layers=1,
        #                    divide_input=False)
        #     self.convs.append(conv)
        #     self.batch_norms.append(BatchNorm(75))

        for _ in range(self.conv_num):
            conv = PNAConv(in_channels=self.in_num, out_channels=self.in_num,
                           aggregators=aggregators, scalers=scalers, deg=_deg,
                           edge_dim=10, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.in_num))

        # self.mlp1 = Sequential(Linear(4, 25), ReLU(), Linear(25, 50), ReLU(), Linear(50, 75))
        # self.mlp2 = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 1))
        self.mlp1 = Sequential(Linear(10, 5), ReLU(), Linear(5, self.in_num))
        self.mlp2 = Sequential(Linear(self.in_num, 5), ReLU(), Linear(5, 1))
        # self.fc = Sequential(Linear(3, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        # print(x)
        # print("x_cp1: {}".format(x.shape))
        # x = self.node_emb(x.squeeze())
        # print("x_cp2: {}".format(x.shape))
        x = self.mlp1(x)
        # print("x_cp3: {}".format(x.shape))
        edge_attr = self.edge_emb(edge_attr)
        """
        x_cp1: torch.Size([8064, 1])
        x_cp2: torch.Size([8064, 75])
        x_cp3: torch.Size([8064, 75])
        x_before: torch.Size([8064, 75])
        x_after: torch.Size([64, 75])
        """
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
        # print("x_before: {}".format(x.shape))
        # print(x)
        # print("batch: {}".format(batch.shape))
        # print(batch)
        x = global_add_pool(x, batch)
        # print("x_after: {}".format(x.shape))
        res = self.mlp2(x)
        return res


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
