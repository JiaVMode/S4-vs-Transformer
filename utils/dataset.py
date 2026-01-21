
import torch
from torch.utils.data import Dataset
import numpy as np

class CharLMDataset(Dataset):
    def __init__(self, data, block_size):
        """
        初始化字符级数据集
        data: 字符串数据 (文本内容)
        block_size: 序列长度 (上下文窗口大小)
        """
        # 构建字符表
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        # 将数据转换为 ID
        self.data_ids = [self.stoi[c] for c in data]
        self.block_size = block_size
        
        # 确定样本数量
        self.n_samples = len(self.data_ids) - block_size
        
    def __len__(self):
        return max(0, self.n_samples)
        
    def __getitem__(self, idx):
        # 切片提取数据
        # 输入 x: data[i : i+L]
        # 目标 y: data[i+1 : i+L+1] (Next Token Prediction)
        
        chunk = self.data_ids[idx : idx + self.block_size + 1]
        chunk = torch.tensor(chunk, dtype=torch.long)
        
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y
