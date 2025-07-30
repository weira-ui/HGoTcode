import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
class PruningAnalyzer:
    def __init__(self):
        """初始化阈值配置"""
        self.threshold_configs = [
            {'S': 0.5, 'type': '宽松阈值', 'score': 78.1, 'time': 12.3},
            {'S': 0.6, 'type': '推荐阈值', 'score': 82.3, 'time': 8.4},
            {'S': 0.7, 'type': '严格阈值', 'score': 73.5, 'time': 6.9},
            {'S': '动态', 'type': '动态阈值', 'score': 81.8, 'time': 9.2}
        ]

    def print_sensitivity_table(self):

        print("剪枝阈值敏感性分析")
        print("| 阈值类型  | 评分(%) | 时间(s) | 检索次数 |")
        print("|------------|---------|---------|----------|")
        print("| 宽松(S=0.5)| 78.1    | 12.3    | 8.2      |")
        print("| 推荐(S=0.6)| 82.3    | 8.4     | 5.7      |")
        print("| 严格(S=0.7)| 73.5    | 6.9     | 3.4      |")
        print("| 动态阈值   | 81.8    | 9.2     | 4.9      |")

    def visualize_tradeoff(self):
        """生成双轴柱状图"""
        plt.rcParams.update({
            'font.sans-serif': ['SimSun'],
            'axes.unicode_minus': False,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelweight': 'bold',
            'legend.fontsize': 10
        })
        # ----------------- 数据准备 -----------------
        thresholds = ['S=0.5', 'S=0.6', 'S=0.7', 'Dynamic']
        factscores = [78.1, 82.3, 73.5, 81.8]  # FactScore百分比
        response_times = [12.3, 8.4, 6.9, 9.2]  # 响应时间（秒）
        retrieval_counts = [8.2, 5.7, 4.1, 6.1]  # 平均检索次数

        # ----------------- 双轴柱状图 -----------------
        # plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})
        fig, ax1 = plt.subplots(figsize=(8, 4.5))

        # 柱状图设置（FactScore）
        bar_width = 0.4
        x = np.arange(len(thresholds))
        bars = ax1.bar(x - bar_width / 2, factscores, bar_width,
                       color=sns.color_palette("Blues", 4),
                       edgecolor='k', linewidth=0.5,
                       label='综合评分', alpha=0.8)

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height - 3,
                     f'{height}%', ha='center', va='top', fontsize=9)

        # 折线图设置（响应时间）
        ax2 = ax1.twinx()
        line = ax2.plot(x + bar_width / 2, response_times,
                        color='#d62728', marker='o', markersize=8,
                        linewidth=2, linestyle='--',
                        label='响应时间')

        # 坐标轴设置
        ax1.set_xticks(x)
        ax1.set_xticklabels(thresholds)
        ax1.set_ylabel('综合评分 (%)', fontsize=12, color='#1f77b4')
        ax2.set_ylabel('响应时间 (秒)', fontsize=12, color='#d62728')
        ax1.set_ylim(60, 90)
        ax2.set_ylim(5, 14)

        # 合并图例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2,
                   loc='upper right', frameon=False)

        plt.title('质量-效率权衡分析', fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig('质量-效率权衡分析.png', dpi=600, bbox_inches='tight')



if __name__ == "__main__":
    analyzer = PruningAnalyzer()

    print("[实验数据表]")
    analyzer.print_sensitivity_table()

    print("\n[可视化示意图]")
    analyzer.visualize_tradeoff()