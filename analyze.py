
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import yaml

# 添加父目录到 sys.path 以便导入 models (如果此脚本在根目录，这行其实不必要，但保留以防万一)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer import Model

# 全局美常设置
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# 加载自定义中文字体
from matplotlib import font_manager
font_path = 'fonts/SimHei.ttf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
 

class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

def load_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)

def plot_training_comparison(save_dir='checkpoints'):
    """
    读取 save_dir 中所有的 *_log.csv 文件并在同一图表中绘制对比。
    """
    if not os.path.exists(save_dir):
        print(f"目录不存在: {save_dir}")
        return

    files = [f for f in os.listdir(save_dir) if f.endswith('_log.csv')]
    
    if not files:
        print(f"在 {save_dir} 中未找到日志文件 (*_log.csv)。")
        return
        
    plt.figure(figsize=(12, 6))
    
    # 子图 1: Loss vs Epoch (损失曲线)
    plt.subplot(1, 2, 1)
    
    for f in files:
        name = f.replace('_log.csv', '')
        # 处理可能的空文件或格式错误
        try:
            df = pd.read_csv(os.path.join(save_dir, f))
            if not df.empty:
                plt.plot(df['epoch'], df['loss'], marker='o', label=name)
        except Exception as e:
            print(f"读取 {f} 失败: {e}")
        
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('训练收敛对比 (Training Convergence)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 2: Time vs Epoch (训练速度)
    plt.subplot(1, 2, 2)
    models = []
    times = []
    
    for f in files:
        name = f.replace('_log.csv', '')
        try:
            df = pd.read_csv(os.path.join(save_dir, f))
            if not df.empty:
                avg_time = df['time'].mean()
                models.append(name)
                times.append(avg_time)
        except:
            pass
        
    plt.bar(models, times, color=['blue', 'orange', 'green'][:len(models)])
    plt.ylabel('Seconds per Epoch')
    plt.title('训练速度 (Training Speed) - 越低越好')
    plt.grid( axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = 'results/comparison_result.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path)
    print(f"已保存对比图至 {output_path}")

def visualize_s4_kernel(config_path, checkpoint_path):
    """
    可视化学习到的 S4 卷积核 (脉冲响应 Impulse Response)。
    """
    if not os.path.exists(checkpoint_path):
        print(f"未找到检查点: {checkpoint_path}")
        return

    config = load_config(config_path)
    # 强制使用 CPU 进行可视化，避免显存问题
    config.device = 'cpu' 
    
    model = Model(config)
    
    # 加载权重
    try:
        # 兼容只保存了 state_dict 或完整 checkpoint 的情况
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"加载 S4 模型失败: {e}")
        return
        
    print("已加载 S4 模型用于 Kernel 可视化。")
    
    # 查找模型中的第一个 S4 Kernel
    s4_kernel = None
    for name, module in model.named_modules():
        if 'kernel' in name and hasattr(module, 'log_dt'):
            s4_kernel = module
            break
            
    if s4_kernel is None:
        print("在模型中未找到 S4 Kernel。")
        return
        
    # 生成 Kernel (脉冲响应)
    L = 1000 # 可视化前 1000 步
    with torch.no_grad():
        k = s4_kernel(L) # (d_model, L)
        
    k_numpy = k.detach().numpy()
    
    plt.figure(figsize=(10, 4))
    # 绘制前几个通道
    for i in range(min(5, k_numpy.shape[0])):
        plt.plot(k_numpy[i], label=f'Channel {i}', alpha=0.7)
        
    plt.title(f"S4 学习到的卷积核 (脉冲响应) - 前 {L} 步")
    plt.xlabel("时间步 (Time Step)")
    plt.ylabel("值 (Value)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = 'results/s4_kernel_viz.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path)
    print(f"已保存 S4 Kernel 可视化至 {output_path}")

def main():
    # 1. 比较训练日志
    plot_training_comparison()
    
    # 2. 可视化 S4 Kernel (如果可用)
    # 假设 config_s4.yaml 和 checkpoints/s4_final.pt 存在
    s4_ckpt = 'checkpoints/s4_final.pt'
    s4_config = 'configs/config_s4.yaml'
    
    if os.path.exists(s4_ckpt) and os.path.exists(s4_config):
        visualize_s4_kernel(s4_config, s4_ckpt)
    else:
        # 尝试查找 epoch 检查点
        import glob
        ckpts = glob.glob('checkpoints/s4_epoch_*.pt')
        if ckpts and os.path.exists(s4_config):
             # 使用最新的一个
             ckpts.sort(key=os.path.getmtime)
             visualize_s4_kernel(s4_config, ckpts[-1])

if __name__ == '__main__':
    main()
