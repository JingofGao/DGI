import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader


def degree(edge_index, N):
    deg = torch.zeros(N)
    for ele in edge_index:
        deg[ele[0]] += 1
    return deg


class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        N, C = x.shape
        deg = degree(edge_index, N)
        A = torch.sparse_coo_tensor(indices=edge_index, values=torch.ones(edge_index.shape[1]).cuda(), size=[N, N])
        D = torch.sparse_coo_tensor(indices=torch.LongTensor([range(N), range(N)]).cuda(), values=deg.cuda(), size=[N, N])
        I = torch.sparse_coo_tensor(indices=torch.LongTensor([range(N), range(N)]).cuda(),
                                    values=torch.ones(N).cuda(), size=[N, N])

        A_tilde = A + I
        D_tilde = D + I
        x = torch.matmul(D_tilde ** -0.5, x)
        x = torch.matmul(A_tilde, x)
        x = torch.matmul(D_tilde ** -0.5, x)
        x = torch.matmul(x, self.W.weight.T)
        return x


class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

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
    dataset_name = "Citeseer"  # Cora or Citeseer
    plot_train_loss = False

    # 加载数据集
    dataset = Planetoid(root='../../data', name=dataset_name)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    # 设置随机种子
    torch.manual_seed(0)

    # 初始化模型和优化器
    model = GCN(input_channels=dataset[0].x.shape[-1],
                hidden_channels=hidden_channels,
                num_classes=len(set(dataset[0].y))).cuda()
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
    print("%s (%s), acc: %.4f" % ("GCN", dataset_name, best_acc))


# Results
# GCN (Cora), acc: 0.8110
# GCN (Citeseer), acc: 0.6870
