import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SGRNAEncoder(nn.Module):
    def __init__(self, embed_dim=4, hidden_dim=16, num_heads=4, lambda_prior=0.5):
        super(SGRNAEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.lambda_prior = lambda_prior  # 生物先验权重

        self.base_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(embed_dim, hidden_dim // num_heads, lambda_prior)
            for _ in range(num_heads)
        ])

        self.out_layer = nn.Linear(hidden_dim, hidden_dim)

    def one_hot_encode(self, seq):
        encoded = torch.zeros(len(seq), self.embed_dim)
        for i, base in enumerate(seq):
            if base in self.base_dict:
                encoded[i, self.base_dict[base]] = 1
        return encoded

    def build_adj_matrix(self, seq):
        n = len(seq)
        adj = torch.zeros(n, n)

        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

        for i in range(n):
            for j in range(n):
                if i != j and seq[j] == complement.get(seq[i], None):
                    adj[i, j] = 1

        return adj

    def forward(self, seq):
        features = self.one_hot_encode(seq)
        adj = self.build_adj_matrix(seq)

        attn_outputs = []
        for attn_head in self.gat_layers:
            attn_outputs.append(attn_head(features, adj, seq))

        h = torch.cat(attn_outputs, dim=1)
        encoded = self.out_layer(h)
        return encoded, adj


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, lambda_prior):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_prior = lambda_prior

        self.W = nn.Linear(input_dim, output_dim)
        self.a = nn.Linear(2 * output_dim, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj, seq):
        Wh = self.W(h)

        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(self.a(a_input).squeeze(2))

        # 计算生物先验分数
        bio_prior = self._calculate_bio_prior(seq, h.shape[0])

        # 合并注意力分数和生物先验
        combined_scores = e + self.lambda_prior * bio_prior

        zero_vec = -9e15 * torch.ones_like(combined_scores)
        attention = torch.where(adj > 0, combined_scores, zero_vec)
        attention = F.softmax(attention, dim=1)

        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)

    def _calculate_bio_prior(self, seq, n_nodes):
        """计算BENodesScore生物先验分数"""
        prior = torch.zeros(n_nodes, n_nodes)

        # 碱基互补强度（AT=0.8, GC=1.0）
        base_strength = {
            ('A', 'T'): 0.8,
            ('T', 'A'): 0.8,
            ('C', 'G'): 1.0,
            ('G', 'C'): 1.0
        }

        # 位置距离衰减（示例：距离越远，影响越小）
        def distance_decay(dist):
            return 1.0 / (1.0 + abs(dist))

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    # 获取碱基对
                    base_pair = (seq[i], seq[j])
                    # 计算位置距离
                    dist = j - i

                    # 计算先验分数
                    if base_pair in base_strength:
                        strength = base_strength[base_pair]
                        decay = distance_decay(dist)
                        prior[i, j] = strength * decay

        return prior

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size(0)

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_pairs_combinations = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_pairs_combinations.view(N, N, 2 * self.output_dim)