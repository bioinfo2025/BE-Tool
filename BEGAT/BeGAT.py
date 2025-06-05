import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple

from BaseEditorEncoder import BaseEditorEncoder
from SGRNAEncoder import SGRNAEncoder


class PositionalEncoding(nn.Module):
    """位置编码层 - 增加最大长度以支持更长序列"""

    def __init__(self, d_model: int, max_len: int = 30):  # 从5000减少到30，覆盖20~23长度
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """确保只使用序列长度部分的位置编码"""
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :]


class TransformerBlock(nn.Module):
    """Transformer编码块（保持不变）"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0
    ):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparseAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = self.attn(self.norm1(x), mask)
        x = x + attn_output
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        return x


class SparseAttention(nn.Module):
    """稀疏自注意力模块（保持不变）"""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0
    ):
        super(SparseAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            bool_mask = mask.to(torch.bool)  # 确保掩码是布尔型
            attn = attn.masked_fill(~bool_mask.unsqueeze(1), -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BeGAT(nn.Module):
    def __init__(self, sgrna_encoder, base_editor_encoder,
                 hidden_dim=64, num_transformer_layers=4,
                 num_attention_heads=8, mlp_dim=128, dropout=0.1):
        super(BeGAT, self).__init__()

        self.sgrna_encoder = sgrna_encoder
        self.base_editor_encoder = base_editor_encoder

        # 特征融合层
        self.feature_projection = nn.Linear(
            sgrna_encoder.hidden_dim + 4,  # 编辑器输出维度固定为4（独热编码）
            hidden_dim
        )

        # 位置编码层 - 支持最大30个碱基（覆盖20~23）
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=30)

        # Transformer编码块
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_attention_heads,
                mlp_ratio=mlp_dim / hidden_dim,
                drop=dropout,
                attn_drop=dropout
            )
            for _ in range(num_transformer_layers)
        ])

        # 输出层 - 预测整体编辑率
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.edge_weights = None  # 可训练的边权重

    def forward(self, sgrna_seq, target_sequence=None):
        """修改：支持20~23个碱基的序列"""
        # 处理目标序列
        if target_sequence is None:
            target_sequence = sgrna_seq

        # 验证序列长度（20~23）
        seq_len = len(sgrna_seq)
        if seq_len < 20 or seq_len > 23:
            raise ValueError(f"序列长度必须在20~23之间，当前长度为{seq_len}")
        if len(target_sequence) != seq_len:
            raise ValueError("sgRNA序列与目标序列长度必须一致")

        # sgRNA编码
        sgrna_features, base_pair_adj = self.sgrna_encoder(sgrna_seq)  # [seq_len, hidden_dim]

        # 碱基编辑器编码
        editor_features = self.base_editor_encoder.encode_sequence(target_sequence)  # [seq_len, 4]

        # 特征融合
        combined_features = torch.cat([sgrna_features, editor_features], dim=1)
        projected_features = self.feature_projection(combined_features)  # [seq_len, hidden_dim]

        # 位置编码
        encoded_features = self.positional_encoding(projected_features.unsqueeze(0)).squeeze(0)

        # 获取编辑窗口掩码
        edit_mask = self.base_editor_encoder.get_edit_window_mask(seq_len)

        # Transformer处理
        x = encoded_features.unsqueeze(0)  # [1, seq_len, hidden_dim]
        for layer in self.transformer_layers:
            x = layer(x, mask=edit_mask.unsqueeze(0).unsqueeze(0))  # 掩码形状 [1, 1, seq_len]

        # 聚合窗口内特征，预测整体编辑率
        window_mask = edit_mask.unsqueeze(-1).expand_as(x)  # [1, seq_len, hidden_dim]
        window_features = x * window_mask  # 仅保留窗口内的特征
        window_avg_features = window_features.sum(1) / (window_mask.sum(1) + 1e-8)  # 平均

        # 预测整体编辑率
        edit_rate = self.output_head(window_avg_features).squeeze()  # 标量值

        return edit_rate, base_pair_adj

def load_begat_model(pkl_path: str, example_editor: BaseEditorEncoder):
    """从pkl文件加载模型"""
    with open(pkl_path, 'rb') as f:
        model_data = pickle.load(f)

    # 重建模型
    sgrna_encoder = model_data['sgrna_encoder']
    model = model_data['model_class'](
        sgrna_encoder=sgrna_encoder,
        base_editor_encoder=example_editor,
        **model_data['model_args']
    )

    # 加载权重
    model.load_state_dict(model_data['state_dict'])
    return model


def predict_edit_rate(model: BeGAT, sgrna: str, target: str,
                      editor_type: str, window_start: int, window_end: int,
                      target_base: str = None, converted_base: str = None):
    """
    使用模型预测编辑率，支持自定义碱基编辑器

    参数:
        model: BeGAT模型
        sgrna: sgRNA序列
        target: 目标DNA序列
        editor_type: 编辑器类型，如"CBE"、"ABE"或自定义类型
        window_start: 编辑窗口起始位置
        window_end: 编辑窗口结束位置
        target_base: 自定义编辑器的目标碱基 (仅当editor_type为自定义类型时需要)
        converted_base: 自定义编辑器的转换碱基 (仅当editor_type为自定义类型时需要)
    """
    # 检查是否需要自定义编辑器
    is_custom_editor = editor_type.upper() not in BaseEditorEncoder.builtin_editors

    # 验证自定义编辑器参数
    if is_custom_editor:
        if not (target_base and converted_base):
            raise ValueError(f"自定义编辑器类型 '{editor_type}' 需要提供 target_base 和 converted_base 参数")
        if target_base not in ['A', 'T', 'C', 'G'] or converted_base not in ['A', 'T', 'C', 'G']:
            raise ValueError("target_base 和 converted_base 必须是A、T、C、G中的一个")

    # 创建编辑器编码器
    editor_encoder = BaseEditorEncoder(
        editor_type=editor_type,
        window_start=window_start,
        window_end=window_end,
        target_base=target_base,
        converted_base=converted_base
    )

    # 更新模型的编辑器编码器
    model.base_editor_encoder = editor_encoder

    # 预测
    model.eval()
    with torch.no_grad():
        pred_rate, _ = model(sgrna, target)

    return pred_rate.item()


def create_dummy_data(num_samples: int = 1000) -> List[Dict]:
    """创建模拟训练数据"""
    data = []
    bases = ['A', 'T', 'C', 'G']
    editor_types = ['CBE', 'ABE']

    for _ in range(num_samples):
        # 随机生成20~23长度的序列
        seq_len = random.randint(20, 23)
        sgrna = ''.join(random.choices(bases, k=seq_len))
        target = ''.join(random.choices(bases, k=seq_len))

        # 随机选择编辑器类型
        editor_type = random.choice(editor_types)

        # 随机生成编辑窗口（确保在序列范围内）
        window_size = random.randint(3, 7)
        window_start = random.randint(0, seq_len - window_size)
        window_end = window_start + window_size - 1

        # 模拟编辑率（窗口中心位置附近更高）
        center_pos = (window_start + window_end) // 2
        edit_rate = 0.8 * (1 - abs(seq_len / 2 - center_pos) / seq_len)
        edit_rate += random.uniform(-0.1, 0.1)  # 添加噪声
        edit_rate = max(0.0, min(1.0, edit_rate))  # 确保在0~1之间

        data.append({
            "sgrna": sgrna,
            "target": target,
            "editor_type": editor_type,
            "window_start": window_start,
            "window_end": window_end,
            "edit_rate": edit_rate
        })

    return data

# 验证不同长度序列的示例
if __name__ == "__main__":
    # 创建编码器
    '''
    sgrna_encoder = SGRNAEncoder(embed_dim=4, hidden_dim=32, num_heads=4)

    # 测试不同长度的编辑窗口
    for seq_len in [20, 21, 22, 23]:
        # 创建对应的编辑窗口（假设窗口在序列中间）
        window_start = max(0, seq_len // 2 - 2)
        window_end = min(seq_len - 1, seq_len // 2 + 2)
        editor_encoder = BaseEditorEncoder(
            editor_type="CBE",
            window_start=window_start,
            window_end=window_end
        )

        # 创建BeGAT模型
        model = BeGAT(
            sgrna_encoder=sgrna_encoder,
            base_editor_encoder=editor_encoder,
            hidden_dim=64,
            num_transformer_layers=3,
            num_attention_heads=4
        )

        # 生成对应长度的序列
        sgrna_seq = "A" * seq_len
        target_seq = "A" * seq_len

        # 前向传播
        edit_rate, adj = model(sgrna_seq, target_seq)
        print(f"序列长度 {seq_len}: 编辑率 = {edit_rate.item():.4f}")
    '''
    # 测试模型加载和预测
    print("\n测试模型加载和预测:")
    example_editor = BaseEditorEncoder(editor_type="CBE", window_start=5, window_end=8)
    loaded_model = load_begat_model('begat_model.pkl', example_editor)
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    loaded_model = loaded_model.to(device)

    test_sample = create_dummy_data(1)[0]
    sgrna = test_sample["sgrna"]
    target = test_sample["target"]
    editor_type = test_sample["editor_type"]
    window_start = test_sample["window_start"]
    window_end = test_sample["window_end"]
    true_rate = test_sample["edit_rate"]

    pred_rate = predict_edit_rate(loaded_model, sgrna, target, editor_type, window_start, window_end)

    print(f"序列: {sgrna}")
    print(f"编辑器类型: {editor_type}, 窗口: [{window_start}, {window_end}]")
    print(f"真实编辑率: {true_rate:.4f}, 预测编辑率: {pred_rate:.4f}")
