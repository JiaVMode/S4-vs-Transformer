
import os
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.transformer import Model
from utils.dataset import CharLMDataset
import urllib.request
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# 全局中文显示支持
plt.rcParams['axes.unicode_minus'] = False 

# 加载自定义中文字体
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

def get_data(config):
    # 检查数据是否存在
    if not os.path.exists(config.data_path):
        os.makedirs(os.path.dirname(config.data_path), exist_ok=True)
        print(f"配置文件中的路径 {config.data_path} 未找到文件。正在下载 TinyShakespeare 数据集...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            urllib.request.urlretrieve(url, config.data_path)
            print("下载完成。")
        except Exception as e:
            print(f"下载失败: {e}。正在生成随机数据用于测试。")
            with open(config.data_path, 'w') as f:
                import random
                chars = "abcdefghijklmnopqrstuvwxyz "
                data = "".join([random.choice(chars) for _ in range(100000)])
                f.write(data)
    
    with open(config.data_path, 'r') as f:
        text = f.read()
        
    return text

def main():
    parser = argparse.ArgumentParser()
    # 支持可选的 --config 参数和位置参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('config_pos', nargs='?', type=str, default=None, help='配置文件路径 (位置参数)')
    args = parser.parse_args()
    
    config_path = args.config if args.config else args.config_pos
    if not config_path:
        raise ValueError("请通过 --config 参数或直接提供路径来指定配置文件")

    config = load_config(config_path)
    
    # 设置设备
    # 如果 config 指定 cuda 但不可用，回退到 cpu
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，自动回退到 CPU 模式")
        config.device = 'cpu'
        
    device = torch.device(config.device)
    print(f"当前使用设备: {device}")
    
    # 数据加载
    text = get_data(config)
    dataset = CharLMDataset(text, config.block_size)
    config.vocab_size = dataset.vocab_size
    print(f"词表大小 (Vocab size): {config.vocab_size}")
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    # 模型初始化
    model = Model(config).to(device)
        
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型 {config.model_type} 已初始化。总参数量: {total_params}")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 日志设置
    os.makedirs(config.save_dir, exist_ok=True)
    log_file = os.path.join(config.save_dir, f"{config.model_type}_log.csv")
    train_history = []
    
    # 训练循环
    model.train()
    print("开始训练...")
    try:
        for epoch in range(config.epochs):
            total_loss = 0
            start_time = time.time()
            
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                output = model(x) # (B, L, V)
                
                # 展平以便计算 Loss
                B, L, V = output.shape
                loss = criterion(output.reshape(-1, V), y.reshape(-1))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if i % config.log_interval == 0:
                    cur_loss = total_loss / (i + 1)
                    elapsed = time.time() - start_time
                    # 仅每隔 log_interval 打印一次信息
                    if i > 0 or epoch == 0:
                         print(f"| Epoch {epoch+1} | Batch {i}/{len(dataloader)} | Loss {cur_loss:.4f} | Time {elapsed:.2f}s |", flush=True)
                    
            avg_loss = total_loss / len(dataloader)
            epoch_time = time.time() - start_time
            print(f"==> Epoch {epoch+1} 结束 | 平均 Loss {avg_loss:.4f} | 耗时 {epoch_time:.2f}s |")
            
            # 保存历史记录
            train_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'time': epoch_time
            })
            
            # 每个 epoch 保存一次检查点
            ckpt_path = os.path.join(config.save_dir, f"{config.model_type}_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            
            # 实时更新 CSV
            df = pd.DataFrame(train_history)
            df.to_csv(log_file, index=False)
            
            # 实时绘图
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['loss'], marker='o', label=f'{config.model_type} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss - {config.model_type}')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(config.save_dir, f"{config.model_type}_loss.png")
            plt.savefig(plot_path)
            plt.close() # 关闭图形释放内存
            print(f"已更新 Loss 曲线图至 {plot_path}")

    except KeyboardInterrupt:
        print("用户中断训练。正在保存当前状态...")
        
    # 保存最终模型
    final_path = os.path.join(config.save_dir, f"{config.model_type}_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"已保存最终模型至 {final_path}")
    
    # 确保日志保存 (以防循环外中断)
    df = pd.DataFrame(train_history)
    df.to_csv(log_file, index=False)
    print(f"已保存训练日志至 {log_file}")

if __name__ == '__main__':
    main()
