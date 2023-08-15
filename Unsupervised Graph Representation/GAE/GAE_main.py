import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCN_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return self.conv2(x, edge_index)


if __name__ == '__main__':
    # args
    out_channels = 128
    epochs = 200
    dataset_name = "Cora"  # Cora or Citeseer

    # dataset
    dataset = Planetoid(root='../../data', name=dataset_name)
    data = dataset[0]
    train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.05,
                                                        num_test=0.1,
                                                        is_undirected=True,
                                                        split_labels=True,
                                                        add_negative_train_samples=True)(data)
    splits = dict(train=train_data, valid=val_data, test=test_data)

    # model
    model = GAE(GCN_Encoder(dataset.num_features, out_channels)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # train and test
    best_auc = 0.
    best_ap = 0.
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_data = splits['train'].cuda()  # 得到训练数据集
        z = model.encode(train_data.x, train_data.edge_index)  # 将训练集投入encoder
        loss = model.recon_loss(z, train_data.edge_index)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model.encode(train_data.x, train_data.edge_index)
            val_auc, val_ap = model.test(z, splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index)
            test_auc, test_ap = model.test(z, splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index)
            print('Epoch: {:03d}, val_AUC: {:.4f}, val_AP: {:.4f}, test_AUC: {:.4f}, test_AP: {:.4f}'
                  .format(epoch, val_auc, val_ap, test_auc, test_ap))

            if test_auc> best_auc:
                best_auc = test_auc
                best_ap = test_ap

    print("best_auc:",best_auc)
    print("best_ap:", best_ap)