
import torch
import torch.nn as nn
from models.s4_layer import S4Layer

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
    def forward(self, x):
        # x: (B, L, D) (Batch, Length, Dim)
        # 因果掩码 (Causal Masking)
        L = x.size(1)
        # 上三角矩阵设为 -inf，防止看到未来信息
        attn_mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
        
        # 获取 Attention 权重以便可视化
        output, weights = self.mha(x, x, x, attn_mask=attn_mask, need_weights=True)
        self.last_attn_weights = weights # 存储权重用于可视化 (Shape: B, H, L, L)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        # 简单的前馈网络 (FFN)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, layer_type='attention', n_head=8, d_ff=None, dropout=0.0):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.ln1 = nn.LayerNorm(d_model)
        
        # 选择混合层类型 (Mixer)
        if layer_type == 'attention':
            self.mixer = SelfAttention(d_model, n_head, dropout)
        elif layer_type == 's4':
            self.mixer = S4Layer(d_model, dropout)
        else:
            raise ValueError(f"未知的层类型: {layer_type}")
            
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x):
        # Pre-norm 结构 (Norm -> Mixer -> Add)
        residual = x
        x = self.ln1(x)
        x = self.mixer(x)
        x = x + residual
        
        # Norm -> MLP -> Add
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x

class Model(nn.Module):
    def __init__(self, config):
        """
        config 需要包含:
        - vocab_size
        - d_model
        - n_layer
        - model_type: 'transformer' | 's4' | 'hybrid'
        - n_head (用于 attention)
        - d_ff
        - dropout
        """
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1024, config.d_model)) # 简单的可学习位置编码
        # 注意: S4 理论上不需要位置编码，但 Hybrid/Transformer 需要。
        
        self.layers = nn.ModuleList()
        
        for i in range(config.n_layer):
            # 确定层类型
            if config.model_type == 'transformer':
                l_type = 'attention'
            elif config.model_type == 's4':
                l_type = 's4'
            else:
                l_type = 'attention'
                
            self.layers.append(Block(
                d_model=config.d_model,
                layer_type=l_type,
                n_head=config.n_head,
                d_ff=config.d_ff,
                dropout=config.dropout
            ))
            
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def forward(self, x):
        # x: (B, L) (Token 索引)
        B, L = x.shape
        
        x = self.embedding(x)
        
        # 添加位置编码 (如果长度允许)
        if L <= self.pos_encoder.shape[1]:
            # 仅当模型包含 Attention 层时添加位置编码
            # (为了公平对比，S4 版本通常不加，或者都加)
            # 这里按照惯例: 纯 S4 不强制需要，但加了也无妨。
            # 为了严谨，如果 model_type != 's4' 才加
            if self.config.model_type != 's4':
                 x = x + self.pos_encoder[:, :L, :]
        else:
            pass # 超出预设长度，暂不处理位置编码扩展
             
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
