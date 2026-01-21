#!/bin/bash
# S4 vs Transformer 可视化脚本
# 该脚本会自动加载 SimHei 中文字体，并生成以下分析图表：
# 1. 训练动态对比 (Loss, Speed)
# 2. 模型困惑度 (Perplexity)
# 3. S4 卷积核脉冲响应 (Impulse Response)
# 4. Transformer Attention Map
# 5. 预测置信度/不确定性 (Entropy)

python visualize.py --mode all --dir checkpoints
