# 简化版可视化代码
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.plot([0.2,0.4,0.6,0.8], [0.89,0.82,0.73,0.61], 'r--o', label='K-BERT')
plt.plot([0.2,0.4,0.6,0.8], [0.91,0.88,0.85,0.79], 'b-s', label='Ours')
plt.xlabel('Conflict Intensity')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('conflict_stability.pdf', bbox_inches='tight')
plt.savefig('figure4-8.png', bbox_inches='tight')