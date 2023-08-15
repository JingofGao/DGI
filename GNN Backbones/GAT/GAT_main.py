import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader


class GATConv(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, mode="concat"):
        super(GATConv, self).__init__()
        self.mode = mode
        self.num_heads = num_heads
        self.a = nn.ModuleList()
        self.w = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()

        if mode=="concat":
            if out_features % num_heads != 0:
                raise Exception("out_features % num_heads != 0")
            for _ in range(num_heads):
                self.a.append(nn.Linear(1, out_features//num_heads * 2))
                self.w.append(nn.Linear(in_features, out_features//num_heads))
        elif mode=="avg":
            for _ in range(num_heads):
                self.a.append(nn.Linear(1, out_features * 2))
                self.w.append(nn.Linear(in_features, out_features))
        else:
            raise Exception("mode error!")

    def forward(self, x, edge_index):
        hs = []
        for i in range(self.num_heads):
            a = self.a[i]
            w = self.w[i]

            h = torch.matmul(x, w.weight.T)
            N = h.shape[-2]

            h_concat = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=-1)
            e = self.leaky_relu(torch.matmul(h_concat, a.weight)).squeeze()
            A = torch.sparse_coo_tensor(indices=edge_index, values=e, size=[N, N])
            attention = torch.sparse.softmax(A, dim=1)

            h = torch.matmul(attention.to_dense(), h)
            hs.append(h)

        if self.mode=="concat":
            return torch.cat(hs, dim=-1)
        elif self.mode=="avg":
            for i in range(1,len(hs)):
                hs[0] += hs[i]
            return hs[0]/len(hs)


class GAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_channels, hidden_channels, num_heads=4, mode="concat")
        self.conv2 = GATConv(hidden_channels, num_classes, num_heads=4, mode="avg")

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


if __name__ == '__main__':
    # args
    lr = 0.01
    epochs = 200
    batch_size = 1
    hidden_channels = 128
    dataset_name = "Cora"  # Cora or Citeseer
    plot_train_loss = False

    # 加载数据集
    dataset = Planetoid(root='../../data', name=dataset_name)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    # 设置随机种子
    torch.manual_seed(0)

    # 初始化模型和优化器
    model = GAT(input_channels=dataset[0].x.shape[-1],
                hidden_channels=hidden_channels,
                num_classes=dataset[0].y.max()+1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    CE_Loss = torch.nn.CrossEntropyLoss()

    # training
    best_acc = 0.
    train_loss_set = []
    for epoch in range(epochs):
        # 训练模型
        model.train()
        train_loss = []
        for x,edge_index,y,train_mask,val_mask,test_mask,batch,ptr in data_loader:
            optimizer.zero_grad()

            X = x[1].cuda()
            Y = y[1].cuda()
            edge_index = edge_index[1].cuda()
            train_mask = train_mask[1].cuda()

            out = model(X, edge_index)
            loss = CE_Loss(out[train_mask], Y[train_mask])
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        train_loss_set.append(train_loss)

        # 测试模型
        model.eval()
        test_acc = []
        with torch.no_grad():
            for x, edge_index, y, train_mask, val_mask, test_mask, batch, ptr in data_loader:
                X = x[1].cuda()
                Y = y[1].cuda()
                edge_index = edge_index[1].cuda()
                test_mask = test_mask[1].cuda()

                out = model(X, edge_index)
                acc = torch.mean((torch.argmax(out[test_mask], dim=-1) == Y[test_mask]).float())
                test_acc += [acc.item()]*len(test_mask)
            test_acc = np.mean(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc

        # 输出训练信息
        print('Epoch [%d/%d], Loss: %.4f, Test Accuracy: %.2f%%'
              % (epoch + 1, epochs, train_loss, 100 * test_acc))
    print("%s (%s), acc: %.4f" % ("GAT", dataset_name, best_acc))


# Results
# GAT (Cora), acc: 0.7980
# GAT (Citeseer), acc: 0.6680
