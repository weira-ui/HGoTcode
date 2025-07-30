# 修复过程堆叠柱状图
from matplotlib import pyplot as plt
plt.rcParams.update({
    'font.sans-serif': ['SimSun'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelweight': 'bold',
    'legend.fontsize': 10
})
stages = ['路径断裂检测', '主路径检索', '分支扩展', '语义关联', '路径生成']
rd30 = [100, 82.4, 68.9, 75.3, 42.1]
rd50 = [100, 73.6, 59.7, 62.4, 35.8]
rd70 = [100, 65.2, 51.3, 58.1, 28.5]

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(stages, rd30, label='RD30%', color='#2ecc70', alpha=0.8)
ax.bar(stages, rd50, bottom=rd30, label='RD50%', color='#f1c40f', alpha=0.8)
ax.bar(stages, rd70, bottom=[i+j for i,j in zip(rd30,rd50)],
       label='RD70%', color='#e74c3c', alpha=0.8)

ax.set_ylabel('阶段成功率 (%)', fontsize=12)
ax.set_title('知识修复过程效能分布', fontsize=14)
ax.legend(loc='upper right')
plt.savefig('修复过程堆叠柱状图.png', bbox_inches='tight')
plt.close()