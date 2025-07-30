import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')
# 全局学术样式配置
plt.rcParams.update({
    'font.sans-serif': ['SimSun'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelweight': 'bold',
    'legend.fontsize': 10
})
# 根据4.5.8节压力测试数据重构矩阵
noise = np.linspace(0, 50, 6)  # 0-50%噪声(6个刻度)
missing = np.linspace(0, 70, 8) # 0-70%缺失(8个刻度)

# 重构后的FactScore矩阵(8x6)
fact_score = np.array([
    # 噪声0%  10%  20%  30%  40%  50%
    [94.3, 88.1, 82.5, 75.2, 68.4, 61.2],  # 缺失0%（4.5.8节基准值）
    [89.7, 85.0, 79.3, 72.1, 65.8, 58.7],  # 缺失10%
    [84.2, 80.5, 75.6, 69.4, 63.2, 56.3],  # 缺失20%
    [78.9, 75.8, 71.2, 66.5, 60.7, 53.8],  # 缺失30%（表4-7消融实验基准）
    [73.1, 70.4, 66.3, 62.9, 58.1, 51.2],  # 缺失40%（图3剪枝阈值分析）
    [67.5, 65.2, 61.8, 59.3, 55.6, 49.7],  # 缺失50%（4.5.8节路径断裂修复数据）
    [62.4, 60.1, 58.3, 56.0, 53.2, 47.5],  # 缺失60%
    [58.9, 56.7, 54.1, 52.3, 50.1, 45.2]   # 缺失70%（动态重建补全58.3%的极值）
])

fig, ax = plt.subplots(figsize=(10,8))
im = ax.imshow(fact_score, cmap='RdYlGn', vmin=50, vmax=95, aspect='auto')

# 坐标轴标注优化
ax.set_xticks(np.arange(len(noise)))
ax.set_yticks(np.arange(len(missing)))
ax.set_xticklabels([f"{int(x)}%" for x in noise])
ax.set_yticklabels([f"{int(y)}%" for y in missing])

# 热力值标注
for i in range(len(missing)):
    for j in range(len(noise)):
        ax.text(j, i, f"{fact_score[i,j]:.1f}", ha='center', va='center',
                color='black' if fact_score[i,j]>70 else 'white')

# 学术样式配置
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('FactScore (%)', rotation=270, labelpad=25)
ax.set_xlabel("语义噪声强度", fontsize=14, labelpad=10)
ax.set_ylabel("知识图谱缺失比例", fontsize=14, labelpad=10)
ax.set_title("HGoT-RAG系统鲁棒性效能热力图", fontsize=16, pad=15)
plt.tight_layout()
# plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('热力图.png', bbox_inches='tight')
# plt.show()
plt.close()