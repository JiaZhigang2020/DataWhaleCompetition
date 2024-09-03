import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch.optim import Adam
import os
from data import DataFile

# 定义单个图模型
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels=4, out_channels=64)  # 22 是原子类型编码的维度
        self.conv2 = GCNConv(in_channels=64, out_channels=32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.mean(x, dim=0)  # 对所有节点进行池化
        return x

# 定义聚合模型
class AggregatedModel(torch.nn.Module):
    def __init__(self):
        super(AggregatedModel, self).__init__()
        self.target_model = GNNModel()
        self.e3_ligase_model = GNNModel()
        self.protac_model = GNNModel()
        self.fc = torch.nn.Linear(in_features=96, out_features=1)  # 3 * 32

    def forward(self, target_data, e3_ligase_data, protac_data):
        target_out = self.target_model(target_data.x, target_data.edge_index)
        e3_ligase_out = self.e3_ligase_model(e3_ligase_data.x, e3_ligase_data.edge_index)
        protac_out = self.protac_model(protac_data.x, protac_data.edge_index)
        combined = torch.cat([target_out, e3_ligase_out, protac_out], dim=0)
        out = self.fc(combined)
        return F.sigmoid(out)

# 训练函数
def train(model, optimizer, data_list, device):
    model.train()
    total_loss = 0
    for data in data_list:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.target,
                       data.e3_ligase,
                       data.protac)
        loss = F.binary_cross_entropy(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_list)


# 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)


# 加载模型
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 预测函数
def predict(model, data_loader, device):
    model.eval()
    predictions = []
    result = 'uuid,Label\n'
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            data = data.to(device)
            output = model(data.target,
                           data.e3_ligase,
                           data.protac)
            if output > 0.5:
                result += f"{index + 1},{1}\n"
            else:
                result += f"{index + 1},{0}\n"
            predictions.append(output.cpu().numpy())
        with open('/mnt/nas/result.txt', 'w') as f:
            f.writelines(result)
    return predictions


if __name__ == '__main__':
    # 加载数据
    # dataFile = DataFile(file_path='/mnt/nas/Project_document/DataWhaleCompetition/dataset/traindata-new.test3.xlsx',
    #                     batch_size=1, shuffle=True)
    # dataFile = DataFile(file_path='../dataset/traindata-new.xlsx',
    #                     batch_size=1, shuffle=True)
    # data_list = dataFile.get_data_list()
    #
    # 定义模型、优化器和设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = AggregatedModel().to(device)
    # optimizer = Adam(model.parameters(), lr=0.01)
    #
    # best_loss = float('inf')
    best_model_path = '/mnt/nas/best_model.pth'
    #
    # # 训练模型
    # for epoch in range(1, 201):  # 训练200个epoch
    #     loss = train(model, optimizer, data_list, device)
    #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    #
    #     # 保存最优模型
    #     if loss < best_loss:
    #         best_loss = loss
    #         save_model(model, best_model_path)

    dataFile = DataFile(file_path='../dataset/testdata-new2.xlsx', batch_size=1, shuffle=True)
    data_list = dataFile.get_data_list()
    # 加载最优模型进行预测
    model = load_model(AggregatedModel(), best_model_path, device)
    predictions = predict(model, data_list, device)
    print(f'Predictions: {predictions}')
