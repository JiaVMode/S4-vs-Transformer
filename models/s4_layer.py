
import torch
import torch.nn as nn
from models.s4_kernel import S4DKernel

class S4Layer(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.kernel = S4DKernel(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # S4 通常使用 GLU 风格的门控机制或者直接投影
        # 这里我们实现一个简单的线性输出投影，类似于 Self-Attention 中的 O-proj
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        输入 x: (B, L, D)
        """
        B, L, D = x.shape
        
        # 1. 计算卷积核 (D, L)
        # 注意: 如果 L 是固定的，kernel 可以只计算一次并缓存。
        # 但为了支持动态 L，我们在前向传播时计算。
        k = self.kernel(L) # (D, L)
        
        # 2. 通过 FFT 进行卷积
        # 变换维度 x: (B, L, D) -> (B, D, L) 以便进行一维卷积
        x_t = x.transpose(1, 2) # (B, D, L)
        
        # FFT 填充 (Padding)
        # 我们需要填充到 2*L 来执行线性卷积 (避免循环卷积的伪影)
        # 也可以只填充到 L+K 的长度。
        n_fft = 2 * L
        
        k_f = torch.fft.rfft(k, n=n_fft) # (D, L+1)
        x_f = torch.fft.rfft(x_t, n=n_fft) # (B, D, L+1)
        
        # 频域逐元素相乘
        # 将 k_f 广播到 Batch 维度
        y_f = x_f * k_f.unsqueeze(0)
        
        # 逆 FFT 变换回时域
        y = torch.fft.irfft(y_f, n=n_fft) # (B, D, 2*L)
        
        # 截断到原始长度 L (因果卷积)
        y = y[:, :, :L]
        
        # 3. 激活函数与投影
        # S4 通常在这里添加残差连接 'x'，但这属于 Block 的职责。
        # 我们只返回混合后的信号。
        
        y = y.transpose(1, 2) # (B, L, D)
        
        y = self.activation(y)
        y = self.dropout(y)
        y = self.output_linear(y)
        
        return y
