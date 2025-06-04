import torch
import numpy as np


class BaseEditorEncoder:
    def __init__(self, editor_type: str, window_start: int, window_end: int,
                 target_base: str = None, converted_base: str = None):
        """
        初始化碱基编辑器编码器

        参数:
            editor_type: 编辑器类型，如 "CBE", "ABE", 或自定义类型
            window_start: 编辑窗口起始位置（0-based）
            window_end: 编辑窗口结束位置（0-based）
            target_base: 目标碱基（仅当editor_type为自定义类型时需要）
            converted_base: 编辑后的碱基（仅当editor_type为自定义类型时需要）
        """
        self.editor_type = editor_type.upper()
        self.window_start = window_start
        self.window_end = window_end
        self.base_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self.complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

        # 内置编辑器类型
        self.builtin_editors = {
            "CBE": {"target": "C", "conversion": "T"},
            "ABE": {"target": "A", "conversion": "G"}
        }

        # 验证编辑器类型
        if self.editor_type in self.builtin_editors:
            self.target_base = self.builtin_editors[self.editor_type]["target"]
            self.converted_base = self.builtin_editors[self.editor_type]["conversion"]
        else:
            # 自定义编辑器类型
            if not (target_base and converted_base):
                raise ValueError(f"自定义编辑器类型需要指定target_base和converted_base")
            if target_base not in self.base_dict or converted_base not in self.base_dict:
                raise ValueError(f"无效的碱基: 必须是A/T/C/G中的一个")
            self.target_base = target_base
            self.converted_base = converted_base

        # 计算互补链编辑规则
        self.complement_target = self.complement[self.target_base]
        self.complement_converted = self.complement[self.converted_base]

    def encode_sequence(self, target_seq: str) -> torch.Tensor:
        """
        对目标序列进行编码，重点处理编辑窗口及填充区域

        参数:
            target_seq: 目标DNA序列

        返回:
            编码后的张量，形状为 [序列长度, 4]
        """
        seq_len = len(target_seq)
        extended_start = max(0, self.window_start - 1)
        extended_end = min(seq_len - 1, self.window_end + 1)
        encoded = np.zeros((seq_len, 4), dtype=np.float32)

        # 对编辑窗口内的碱基进行编码
        for i in range(seq_len):
            if extended_start <= i <= extended_end:
                base = target_seq[i]
                if base not in self.base_dict:
                    raise ValueError(f"无效碱基: {base}")
                encoded[i, self.base_dict[base]] = 1
            else:
                # 窗口外区域根据编辑器类型填充
                encoded[i, self.base_dict[self.target_base]] = 0.5
                encoded[i, self.base_dict[self.complement_target]] = 0.5

        return torch.tensor(encoded, dtype=torch.float32)

    def get_edit_window_mask(self, sequence_length: int) -> torch.Tensor:
        """获取编辑窗口的掩码"""
        mask = torch.zeros(sequence_length)
        start = max(0, self.window_start - 1)
        end = min(sequence_length - 1, self.window_end + 1)
        mask[start:end + 1] = 1
        return mask