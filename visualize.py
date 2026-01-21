
import os
import sys
import argparse
import yaml
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from models.transformer import Model
import numpy as np

# 全局美学设置
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False 

# 加载自定义中文字体
font_path = 'fonts/SimHei.ttf'
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    # 手动注册字体以防萬一
    font_manager.fontManager.addfont(font_path)
    print(f"已加载中文字体: {font_path}")
else:
    print("未找到中文字体 fonts/SimHei.ttf，可能无法正确显示中文。")


class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

def load_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)

def viz_comparison(save_dir, output_file="results/viz_comparison.png"):
    """
    1. 性能与速度对比
    """
    print(f"从 {save_dir} 生成对比图...")
    if not os.path.exists('results'):
        os.makedirs('results')

    files = [f for f in os.listdir(save_dir) if f.endswith('_log.csv')]
    if not files:
        print("未找到日志文件。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 损失曲线
    colors = sns.color_palette("husl", len(files))
    for i, f in enumerate(files):
        name = f.replace('_log.csv', '').upper()
        df = pd.read_csv(os.path.join(save_dir, f))
        sns.lineplot(data=df, x='epoch', y='loss', label=name, ax=axes[0], marker='o', linewidth=2, color=colors[i])
    
    axes[0].set_title("训练损失对比 (Training Loss)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].legend()

    # 训练速度
    names = []
    speeds = []
    for f in files:
        name = f.replace('_log.csv', '').upper()
        df = pd.read_csv(os.path.join(save_dir, f))
        names.append(name)
        speeds.append(df['time'].mean())
    
    sns.barplot(x=names, y=speeds, ax=axes[1], palette=colors)
    axes[1].set_title("训练速度 (每Epoch耗时)")
    axes[1].set_ylabel("秒 (Seconds)")
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"已保存: {output_file}")

def viz_s4_kernel(config_path, checkpoint_path, output_file="results/viz_s4_kernel.png"):
    """
    2. S4 Kernel 可视化
    """
    print(f"从 {checkpoint_path} 生成 S4 Kernel 可视化...")
    if not os.path.exists('results'):
        os.makedirs('results')
        
    if not os.path.exists(checkpoint_path):
        print(f"未找到检查点 {checkpoint_path}。")
        return

    config = load_config(config_path)
    config.device = 'cpu'
    
    model = Model(config)
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"加载模型错误: {e}")
        return

    # 查找 kernel
    kernel_layer = None
    for m in model.modules():
        if hasattr(m, 'log_dt'): # 识别 S4 Kernel
            kernel_layer = m
            break
            
    if kernel_layer:
        L = 200 # 可视化前 200 步
        with torch.no_grad():
            k = kernel_layer(L).detach().numpy() # (D, L)
            
        plt.figure(figsize=(12, 6))
        # 绘制前 3 个通道
        for i in range(min(3, k.shape[0])):
            plt.plot(k[i], label=f'Feature Channel {i}')
        
        plt.title(f"S4 Kernel 脉冲响应 (前 {L} 步)")
        plt.xlabel("时间延迟 (Time Lag)")
        plt.ylabel("幅度 (Magnitude)")
        plt.legend()
        plt.savefig(output_file)
        print(f"已保存: {output_file}")
    else:
        print("模型中未找到 S4 Kernel。")

def viz_attention(config_path, checkpoint_path, output_file="results/viz_attention.png"):
    """
    3. Attention Map 可视化
    """
    print(f"从 {checkpoint_path} 生成 Attention Map 可视化...")
    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists(checkpoint_path):
        print(f"未找到检查点 {checkpoint_path}。")
        return

    config = load_config(config_path)
    config.device = 'cpu'
    
    model = Model(config)
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
    except:
        print("加载模型错误。")
        return

    # 构造虚拟输入
    L = 50
    x = torch.randint(0, config.vocab_size, (1, L))
    
    with torch.no_grad():
        model(x)
        
    # 查找最后的 attention weights
    attn_weights = None
    for m in model.modules():
        if hasattr(m, 'last_attn_weights') and m.last_attn_weights is not None:
            attn_weights = m.last_attn_weights[0] # (1, H, L, L) -> (H, L, L)
            break
            
        if attn_weights is not None:
        # 绘制第一个 Head
        plt.figure(figsize=(10, 8))
        # 确保是 2D (L, L)
        data = attn_weights[0].numpy().astype(float)
        
        if len(data.shape) == 1:
             # 有时可能被压扁了，尝试 Reshape 或只画一部分
             L = int(np.sqrt(data.shape[0]))
             data = data[:L*L].reshape(L, L)
             
        sns.heatmap(data, cmap="viridis", square=True)
        plt.title("Self-Attention Map (Head 0)")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.savefig(output_file)
        print(f"已保存: {output_file}")
    else:
        print("未找到 Attention Weights (模型可能是纯 S4 或未启用权重保存)。")


def viz_perplexity(save_dir, output_file="results/viz_perplexity.png"):
    """
    4. 困惑度 (Perplexity) 对比
    PPL = exp(CrossEntropyLoss)
    """
    print(f"从 {save_dir} 生成困惑度 (Perplexity) 对比图...")
    if not os.path.exists('results'):
        os.makedirs('results')
        
    files = [f for f in os.listdir(save_dir) if f.endswith('_log.csv')]
    if not files:
        print("未找到日志文件。")
        return
        
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(files))
    
    for i, f in enumerate(files):
        name = f.replace('_log.csv', '').upper()
        df = pd.read_csv(os.path.join(save_dir, f))
        # PPL 计算
        df['ppl'] = np.exp(df['loss'])
        sns.lineplot(data=df, x='epoch', y='ppl', label=name, marker='o', linewidth=2, color=colors[i])
        
    plt.title("模型困惑度对比 (Perplexity) - 越低越好")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.savefig(output_file)
    print(f"已保存: {output_file}")


def viz_confidence(config_path, checkpoint_path, output_file="results/viz_confidence.png"):
    """
    5. 预测置信度 (Confidence/Entropy) 可视化
    展示模型在生成一段文本时，每一步预测分布的信息熵。
    熵越低 -> 模型越确信。
    """
    print(f"从 {checkpoint_path} 生成预测置信度分析图...")
    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists(checkpoint_path):
        print(f"未找到检查点 {checkpoint_path}。")
        return

    config = load_config(config_path)
    config.device = 'cpu'
    model = Model(config)
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
    except:
        print("加载模型错误。")
        return
        
    # 构造一段测试序列
    model.eval()
    L = 100
    x = torch.randint(0, config.vocab_size, (1, L))
    
    with torch.no_grad():
        logits = model(x) # (1, L, V)
        probs = torch.softmax(logits, dim=-1)
        # 计算熵: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).squeeze(0).numpy() # (L,)
        
    plt.figure(figsize=(12, 4))
    plt.plot(entropy, label='Prediction Entropy', color='purple')
    plt.title(f"预测不确定性 (Entropy) - {config.model_type.upper()}")
    plt.xlabel("Sequence Position")
    plt.ylabel("Entropy (Uncertainty)")
    plt.fill_between(range(L), entropy, alpha=0.3, color='purple')
    plt.savefig(output_file)
    print(f"已保存: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="可视化工具")
    parser.add_argument('--mode', type=str, required=True, choices=['all', 'compare', 's4', 'attn', 'extra'], help='可视化模式')
    parser.add_argument('--dir', type=str, default='checkpoints', help='检查点目录')
    args = parser.parse_args()
    
    # 确保 results 目录存在
    if not os.path.exists('results'):
        os.makedirs('results')
    
    if args.mode in ['compare', 'all', 'extra']:
        viz_comparison(args.dir)
        viz_perplexity(args.dir)
        
    if args.mode in ['s4', 'all', 'extra']:
        # 查找 S4 特定文件
        s4_ckpt = os.path.join(args.dir, 's4_final.pt')
        if not os.path.exists(s4_ckpt): s4_ckpt = os.path.join(args.dir, 's4_model.pt')
        viz_s4_kernel("configs/config_s4.yaml", s4_ckpt)
        
        # S4 的置信度
        if os.path.exists(s4_ckpt):
             viz_confidence("configs/config_s4.yaml", s4_ckpt, "results/viz_confidence_s4.png")
        
    if args.mode in ['attn', 'all']:
        # 查找 Transformer 文件
        tf_ckpt = os.path.join(args.dir, 'transformer_final.pt')
        if not os.path.exists(tf_ckpt): tf_ckpt = os.path.join(args.dir, 'transformer_model.pt')
        viz_attention("configs/config_transformer.yaml", tf_ckpt)
        
        if os.path.exists(tf_ckpt):
             viz_confidence("configs/config_transformer.yaml", tf_ckpt, "results/viz_confidence_transformer.png")

if __name__ == '__main__':
    main()
