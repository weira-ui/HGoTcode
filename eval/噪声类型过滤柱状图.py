# 噪声类型过滤柱状图
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
noise_labels = ['拼写错误', '语法混乱', '实体混淆', '逻辑干扰', '混合噪声']
ntr_10 = [92.3, 84.7, 88.9, 76.8, 81.5]
ntr_50 = [78.1, 68.3, 71.3, 62.7, 61.2]

fig, ax = plt.subplots(figsize=(10,6))
x = np.arange(len(noise_labels))
width = 0.35

rects1 = ax.bar(x - width/2, ntr_10, width, label='10%噪声强度', color='#2ecc71')
rects2 = ax.bar(x + width/2, ntr_50, width, label='50%噪声强度', color='#3498db')

ax.set_ylabel('噪声过滤率(NTR) (%)', fontsize=12)
ax.set_title('不同类型噪声的过滤效能对比', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(noise_labels, rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig('噪声类型过滤柱状图.png', bbox_inches='tight')
plt.close()
