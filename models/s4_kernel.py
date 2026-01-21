
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class S4DKernel(nn.Module):
    """
    S4D (Diagonal State Space) Kernel (S4D 核心模块).
    计算由对角矩阵 A 定义的状态空间模型的卷积核。
    
    参考文献:
    - "On the Parameterization and Initialization of Diagonal State Space Models" (Gu et al., 2022)
    """
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.N = N
        
        # 1. 初始化 A (对角系统矩阵)
        # 我们使用 S4D-Lin 初始化: 实部为 -0.5, 虚部均匀分布
        # A = -0.5 + i * 2*pi*n/N
        log_dt = torch.rand(self.d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
        # A_re (实部) 通常固定为 -0.5 以保证 HiPPO-LegS 近似的稳定性
        self.A_re = nn.Parameter(torch.full((self.d_model, self.N), -0.5))
        
        # A_im (虚部): 频率。我们将其初始化为覆盖特定范围。
        # 通常近似为 2*pi * n / N，其中 n 在 [0, N/2] 范围
        # 使用 S4D-Lin 初始化
        freqs = torch.arange(self.N) # 0 到 N-1
        # S4D-Lin: 虚部大概是 n * pi。我们这里使用随机初始化或均匀分布。
        self.A_im = nn.Parameter(torch.arange(self.N).float().repeat(self.d_model, 1))
        
        # C 是输出投影 (复数)
        # 随机正态分布初始化
        self.C = nn.Parameter(torch.randn(self.d_model, self.N, 2)) # (H, N, 实部/虚部)
        
        # 在 S4D 中，B 通常被吸收到 C 中或者固定为 1。
        # 为了灵活性我们可以保留它，但这里我们实际上只用 C 就够了 (有效地令 B=1)。
        
    def forward(self, L):
        """
        计算长度为 L 的卷积核。
        返回: (d_model, L)
        """
        dt = torch.exp(self.log_dt) # (H,)
        
        # 构造 A = A_re + i * A_im
        # 我们需要确保 A_re 是负数以保证稳定性
        A = torch.complex(self.A_re, self.A_im) # (H, N)
        
        # 离散化 A -> A_bar = exp(A * dt)
        # 形状: (H, N)
        dt_A = A * dt.unsqueeze(-1)  # (H, N)
        
        # 根据 S4D 公式计算卷积核 K[t] = C * (A^t) * B
        # 理想情况下是求和: sum_n C_n * exp(A_n * dt * t)
        
        # t: 时间步索引 (L)
        t = torch.arange(L, device=A.device).float()
        
        # 我们可以高效地计算这个求和。
        # rate (速率) = dt * A
        rate = dt_A # (H, N)
        
        # 计算 K[t, h] = \sum_n C[h, n] * exp(rate[h, n] * t)
        
        # 将 C 转为复数张量
        C = torch.view_as_complex(self.C) # (H, N)
        
        # 使用 einsum 进行计算:
        # kt = exp(rate * t)
        # 形状变换: (H, N) x (L) -> (H, N, L)
        kt = torch.einsum("hn, l -> hnl", rate, t) # (H, N, L) 复数
        kt = torch.exp(kt)
        
        # 将 C 与 exp 项相乘并求和 (对 N 维度求和)
        # K = sum over n
        K = torch.einsum("hn, hnl -> hl", C, kt) # (H, L)
        
        return K.real # 返回脉冲响应的实部 (因为 S4 输出的是实信号)
