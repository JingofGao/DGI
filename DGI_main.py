import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax
import warnings
warnings.filterwarnings("ignore")


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


if __name__ == '__main__':
    # args
    out_channels = 128
    epochs = 200
    dataset_name = "Citeseer"  # Cora or Citeseer

    # dataset
    dataset = Planetoid(root='../../data', name=dataset_name)
    data = dataset[0]

    # model
    model = DeepGraphInfomax(
        hidden_channels=out_channels,
        encoder=Encoder(in_channels=dataset.num_features, out_channels=out_channels),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train and test
    best_acc = 0.
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        data = data.cuda()
        pos_z, neg_z, summary = model(data.x, data.edge_index)  # 将训练集投入encoder
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z, _, _ = model(data.x, data.edge_index)
            test_acc = model.test(train_z=z[data.train_mask], train_y=data.y[data.train_mask], test_z=z[data.test_mask],
                             test_y=data.y[data.test_mask])
            if test_acc> best_acc:
                best_acc = test_acc
        print('Epoch: {:03d}, Loss: {:.4f}, test_acc: {:.4f}' .format(epoch, loss.item(), test_acc))

    print("best_acc:", best_acc)


# Cora dataset: 0.8090
# Citeseer dataset: 0.7010
