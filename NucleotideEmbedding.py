import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class NucleotideEmbedding(nn.Module):
    """核苷酸嵌入层"""

    def __init__(self, embedding_dim=64):
        super().__init__()
        # A, T, C, G, N(未知)
        self.embedding = nn.Embedding(5, embedding_dim)

    def forward(self, x):
        # 输入x: (batch_size, seq_len) 索引形式序列
        return self.embedding(x)  # (batch_size, seq_len, embed_dim)


class DynamicWindowEncoder(nn.Module):
    """动态窗口编码模块"""

    def __init__(self, window_sizes=[3, 5, 7], embed_dim=64):
        super().__init__()
        self.window_sizes = window_sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=w, padding=w // 2)
            for w in window_sizes
        ])
        self.attention = nn.Linear(embed_dim, len(window_sizes))

    def forward(self, x):
        """
        输入x: (batch_size, seq_len, embed_dim)
        输出: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 多尺度卷积特征提取
        conv_features = []
        x_perm = x.permute(0, 2, 1)  # (batch, embed, seq)
        for conv in self.convs:
            f = conv(x_perm).permute(0, 2, 1)  # (batch, seq, embed)
            conv_features.append(f)

        # 动态注意力权重
        attn_weights = F.softmax(self.attention(x), dim=-1)  # (batch, seq, n_windows)

        # 加权融合
        combined = torch.stack(conv_features, dim=3)  # (batch, seq, embed, n_windows)
        attn_weights = attn_weights.unsqueeze(2)  # (batch, seq, 1, n_windows)
        weighted = (combined * attn_weights).sum(dim=3)

        return weighted + x  # 残差连接


class GATModule(nn.Module):
    """图注意力网络模块"""

    def __init__(self, embed_dim=64, heads=4):
        super().__init__()
        # 定义GAT层
        self.gat1 = GATConv(embed_dim, embed_dim, heads=heads)
        self.gat2 = GATConv(embed_dim * heads, embed_dim, heads=1)

    def forward(self, x, edge_index):
        """
        输入:
            x: 节点特征 (num_nodes, embed_dim)
            edge_index: 边索引 (2, num_edges)
        输出: 更新后的节点特征 (num_nodes, embed_dim)
        """
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x


class HybridModel(nn.Module):
    """整合GAT、动态窗口和Transformer的混合模型"""

    def __init__(self, embed_dim=64, transformer_layers=3):
        super().__init__()
        # 序列嵌入
        self.embedding = NucleotideEmbedding(embed_dim)

        # 动态窗口编码
        self.window_encoder = DynamicWindowEncoder(embed_dim=embed_dim)

        # 图注意力网络
        self.gat = GATModule(embed_dim)

        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_dim * 2,  # 拼接GAT和窗口特征
            nhead=4,
            dim_feedforward=256
        )
        self.transformer = TransformerEncoder(encoder_layers, transformer_layers)

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, seq_data, graph_data):
        """
        输入:
            seq_data: 序列数据 (batch, seq_len)
            graph_data: 图数据 (x, edge_index)
        输出: 预测概率 (batch, 1)
        """
        # --------------------------
        # 分支1: 动态窗口编码
        # --------------------------
        embedded = self.embedding(seq_data)  # (batch, seq, embed)
        window_feat = self.window_encoder(embedded)  # (batch, seq, embed)

        # --------------------------
        # 分支2: 图注意力网络
        # --------------------------
        x, edge_index = graph_data
        gat_feat = self.gat(x, edge_index)  # (num_nodes, embed)
        gat_feat = gat_feat.view(seq_data.size(0), -1, gat_feat.size(-1))  # (batch, seq, embed)

        # --------------------------
        # 特征融合
        # --------------------------
        combined = torch.cat([window_feat, gat_feat], dim=-1)  # (batch, seq, embed*2)

        # --------------------------
        # Transformer编码
        # --------------------------
        # 调整维度: (seq, batch, embed)
        trans_input = combined.permute(1, 0, 2)
        trans_output = self.transformer(trans_input)  # (seq, batch, embed*2)
        trans_output = trans_output.permute(1, 0, 2)  # (batch, seq, embed*2)

        # --------------------------
        # 全局平均池化 + 预测
        # --------------------------
        pooled = trans_output.mean(dim=1)  # (batch, embed*2)
        return self.predictor(pooled)


# --------------------------
# 示例用法
# --------------------------
if __name__ == "__main__":
    # 超参数
    batch_size = 8
    seq_len = 20
    embed_dim = 64

    # 模拟输入数据
    seq_input = torch.randint(0, 4, (batch_size, seq_len))  # 序列数据
    node_features = torch.randn(batch_size * seq_len, embed_dim)  # 假设每个核苷酸是节点
    edge_index = torch.randint(0, seq_len, (2, 30))  # 随机生成边

    # 初始化模型
    model = HybridModel(embed_dim=embed_dim)

    # 前向传播
    output = model(seq_input, (node_features, edge_index))
    print("预测结果形状:", output.shape)
    # 输出: torch.Size([8, 1])