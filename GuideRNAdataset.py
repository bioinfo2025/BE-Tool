import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle


# 假设已定义前文的 HybridModel 类
# from model import HybridModel

# 自定义数据集类（需根据实际数据修改）
from NucleotideEmbedding import HybridModel


class GuideRNAdataset(Dataset):
    def __init__(self, seq_data, graph_data, labels):
        """
        参数:
            seq_data: 序列数据 (num_samples, seq_len)
            graph_data: 图数据元组 (node_features, edge_index)
            labels: 标签数据 (num_samples,)
        """
        self.seq_data = seq_data
        self.node_features = graph_data[0]
        self.edge_index = graph_data[1]
        self.labels = labels

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        return (
            self.seq_data[idx],
            (self.node_features[idx], self.edge_index),
            self.labels[idx]
        )


# 训练函数
def train_model(model, dataloader, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()  # 二分类任务使用交叉熵
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for seq, graph, labels in dataloader:
            seq = seq.to(device)
            graph = (graph[0].to(device), graph[1].to(device))
            labels = labels.float().to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(seq, graph)
            loss = criterion(outputs.squeeze(), labels)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 打印统计信息
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f}")

    return model


if __name__ == "__main__":
    # 假设参数
    batch_size = 32
    seq_len = 20
    embed_dim = 64
    num_samples = 1000  # 示例数据量

    # 生成模拟数据（实际应替换为真实数据）
    # 序列数据: 整数编码 (0-4 表示 A/T/C/G/N)
    seq_data = torch.randint(0, 4, (num_samples, seq_len))
    # 图节点特征: (num_samples*seq_len, embed_dim)
    node_features = torch.randn(num_samples * seq_len, embed_dim)
    # 边索引: 随机生成示例边 (实际应根据互补配对等规则构建)
    edge_index = torch.randint(0, seq_len, (2, 50))
    # 标签: 随机生成0/1标签
    labels = torch.randint(0, 2, (num_samples,)).float()

    # 创建数据集和数据加载器
    dataset = GuideRNAdataset(seq_data, (node_features, edge_index), labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = HybridModel(embed_dim=embed_dim)

    # 训练模型
    trained_model = train_model(model, dataloader, epochs=50, lr=0.001)

    # 保存整个模型为 .pkl 文件（包含结构和参数）
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(trained_model, f)

    # 保存方式2: 官方推荐方法（需加载时提供模型类定义）
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'embed_dim': embed_dim
    }, "trained_model_checkpoint.pth")

    print("模型已保存为 trained_model.pkl 和 trained_model_checkpoint.pth")