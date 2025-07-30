import matplotlib.pyplot as plt
import numpy as np
# 全局学术样式配置
plt.rcParams.update({
    'font.sans-serif': ['SimSun'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelweight': 'bold',
    'legend.fontsize': 10
})
# 数据准备
thresholds = [0.5, 0.6, 0.7]
retrieval_counts = [8.2, 5.7, 6.9]  # 平均检索次数
capture_rates = [
    [54.3, 78.6, 91.2],  # S=0.5时各次检索累计捕获率
    [68.9, 89.5, 98.2],  # S=0.6时累计捕获率
    [42.1, 63.7, 76.4]   # S=0.7时累计捕获率
]

# 创建画布
plt.figure(figsize=(12, 6), dpi=300)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示

# 主图：检索次数折线图
ax1 = plt.subplot(1, 2, 1)
line = ax1.plot(thresholds, retrieval_counts,
               marker='o', color='#2c7bb6',
               linewidth=2, markersize=8,
               label='平均检索次数')

# 标注关键数据点
ax1.annotate(f'{retrieval_counts[0]}次', xy=(0.5,8.2),
            xytext=(0.45,8.5), arrowprops=dict(arrowstyle='->'))
ax1.annotate(f'-30.5%', xy=(0.6,5.7),
            xytext=(0.55,5.0), arrowprops=dict(arrowstyle='->'))

# 帕累托最优区间标注
ax1.axvspan(0.58, 0.62, alpha=0.2, color='gold',
           label='帕累托最优区间')

# 坐标轴设置
ax1.set_xlabel('剪枝阈值 (S)', fontsize=12)
ax1.set_ylabel('平均检索次数', color='#2c7bb6', fontsize=12)
ax1.tick_params(axis='y', colors='#2c7bb6')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_title('(a) 阈值与检索行为关系', fontsize=13, pad=10)

# 副图：捕获效率条形图
ax2 = plt.subplot(1, 2, 2)
bar_width = 0.25
colors = ['#d7191c', '#fdae61', '#abd9e9']

# 各阈值下的累计捕获率展示
for i, s in enumerate(thresholds):
    positions = np.arange(3) + i*bar_width
    ax2.bar(positions, capture_rates[i],
           width=bar_width, color=colors[i],
           label=f'S={s}')

# 添加基准线
ax2.axhline(98.2, xmin=0.05, xmax=0.95,
          color='#2c7bb6', linestyle='--',
          linewidth=1.5, alpha=0.7)

# 标注最优结果
ax2.text(0.8, 99, f'98.2% @ S=0.6',
        color='#2c7bb6', fontsize=10)

# 坐标轴设置
ax2.set_xticks([0.25, 1.25, 2.25])
ax2.set_xticklabels(['第1次检索', '第2次检索', '第3次检索'])
ax2.set_ylabel('累计有效知识捕获率 (%)', fontsize=12)
ax2.set_ylim(0, 110)
ax2.grid(axis='y', linestyle='--', alpha=0.6)
ax2.set_title('(b) 检索效率分析', fontsize=13, pad=10)

# 图例与布局优化
ax1.legend(loc='upper right', frameon=False)
ax2.legend(loc='upper left', frameon=False)
plt.tight_layout(pad=3)
plt.savefig('figure4-4.png', bbox_inches='tight')
# plt.show()