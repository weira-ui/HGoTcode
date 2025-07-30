import matplotlib.pyplot as plt
import numpy as np

# 全局字体配置（符合期刊排版要求）
plt.rcParams.update({
    'font.sans-serif': ['SimSun'],  # 正式论文推荐宋体
    'axes.unicode_minus': False,
    'font.size': 12,  # 正文文字大小
    'axes.titlesize': 14,  # 标题字号
    'axes.labelweight': 'bold',  # 坐标轴标签加粗
    'legend.fontsize': 10  # 图例字号
})


def save_figure5_6a():
    """噪声强度影响分析图"""
    # 实验数据（来自表4-7消融实验）
    noise_levels = ['10%', '30%', '50%', '70%']
    hgot_em = [88.4, 76.2, 61.2, 35.7]
    baseline_em = [72.3, 53.1, 27.5, 9.8]

    plt.figure(figsize=(8, 6))
    bar_width = 0.35
    x = np.arange(len(noise_levels))

    # 双柱状图绘制
    rects1 = plt.bar(x - bar_width / 2, hgot_em, bar_width,
                     color='#1f77b4', label='HGoT-RAG')
    rects2 = plt.bar(x + bar_width / 2, baseline_em, bar_width,
                     color='#ff7f0e', label='基线模型')

    # 标注配置
    plt.xticks(x, noise_levels)
    plt.xlabel('噪声强度', fontweight='bold')
    plt.ylabel('精确匹配率 (%)', fontweight='bold')
    plt.title('噪声强度对EM指标的影响', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for rect in rects1 + rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., height,
                 f'{height}%', ha='center', va='bottom')

    plt.legend(loc='upper right')
    plt.savefig('噪声强度影响分析图.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_figure5_6b():
    """剪枝阈值分析图（双Y轴）"""
    # 数据来源：4.5.7节消融实验
    thresholds = ['0.5', '0.6', '0.7', '动态']
    accuracy = [78.1, 82.3, 73.5, 81.8]
    response_time = [12.3, 8.4, 6.9, 9.2]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 柱状图（主Y轴）
    bars = ax1.bar(thresholds, accuracy, color='#2ca02c', alpha=0.8)
    ax1.set_ylabel('准确率 (%)', fontweight='bold')
    ax1.tick_params(axis='y')

    # 折线图（次Y轴）
    ax2 = ax1.twinx()
    line, = ax2.plot(thresholds, response_time, 'D-',
                     color='#d62728', lw=2, markersize=8)
    ax2.set_ylabel('响应时间 (秒)', fontweight='bold')

    # 联合图例
    ax1.legend(
        handles=[bars, line],
        labels=['准确率', '响应时间'],
        bbox_to_anchor=(-0.01, 1.13),
        loc='upper left',
        frameon=True,
        shadow=True
    )

    plt.title('质量-效率权衡分析', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('剪枝阈值分析图.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_figure5_7():
    """知识修复效能分析图"""
    # 数据来源：表4-6检索方法对比
    scenarios = ['RD30%', 'RD50%', 'RD70%', 'AT10%']
    krr = [65.3, 52.7, 42.1, 88.4]
    time = [3.2, 5.1, 9.8, 2.3]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 柱状图（主Y轴）
    bars = ax1.bar(scenarios, krr, color='#9370DB', alpha=0.8)
    ax1.set_ylabel('知识修复率 (%)', fontweight='bold')

    # 折线图（次Y轴）
    ax2 = ax1.twinx()
    line, = ax2.plot(scenarios, time, 's--',
                     color='#FF6347', lw=2, markersize=8)
    ax2.set_ylabel('响应时间 (秒)', fontweight='bold')

    # 联合图例
    ax1.legend([bars, line], ['知识修复率', '响应时间'],bbox_to_anchor=(0.21, 1.0),
               loc='upper right', frameon=True)

    plt.title('知识修复效能分析', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('知识修复效能分析图.png', dpi=300, bbox_inches='tight')
    plt.close()


# 执行绘图（符合4.5.4节实验流程）
if __name__ == "__main__":
    save_figure5_6a()
    save_figure5_6b()
    save_figure5_7()