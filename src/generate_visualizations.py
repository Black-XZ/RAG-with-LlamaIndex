"""
生成四组实验对比可视化图表
=================================
使用方法: python src/generate_visualizations.py
输出目录: results/runs/analysis/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 数据路径
DATA_DIR = Path("results/runs")

# 加载评分后的数据
configs = {
    "Flan-T5/256": DATA_DIR / "eval_flant5_256/eval_flant5_chunk256_scored.csv",
    "Flan-T5/512": DATA_DIR / "eval_flant5_512/eval_flant5_chunk512_scored.csv",
    "Mistral-7B/256": DATA_DIR / "eval_mistral7b_256/eval_mistral7b_chunk256_scored.csv",
    "Mistral-7B/512": DATA_DIR / "eval_mistral7b_512/eval_mistral7b_chunk512_scored.csv",
}

print("加载数据...")
dfs = {}
for name, path in configs.items():
    if path.exists():
        dfs[name] = pd.read_csv(path)
        print(f"  [{name}] {len(dfs[name])} 条记录")
    else:
        print(f"  [警告] 文件不存在: {path}")

# 创建输出目录
output_dir = DATA_DIR / "analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# 颜色配置
colors_flant5 = ['#3498db', '#2980b9']  # 蓝色系
colors_mistral = ['#27ae60', '#1e8449']  # 绿色系
colors = [colors_flant5[0], colors_flant5[1], colors_mistral[0], colors_mistral[1]]

print("\n生成可视化图表...")

# ============== 图1: Relevance Score 对比 ==============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 柱状图
relevance_means = [dfs[name]['relevance_score'].mean() for name in configs]
relevance_stds = [dfs[name]['relevance_score'].std() for name in configs]

bars = axes[0].bar(list(configs.keys()), relevance_means, yerr=relevance_stds,
                    capsize=5, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

axes[0].set_ylabel('Relevance Score (1-5)', fontsize=12)
axes[0].set_title('Average Relevance Score by Configuration', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 5.8)
axes[0].axhline(y=3, color='red', linestyle='--', alpha=0.5, label='Baseline (3)')
axes[0].legend(loc='upper left')

# 添加数值标签
for bar, mean, std in zip(bars, relevance_means, relevance_stds):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.15,
                  f'{mean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 箱线图
relevance_data = [dfs[name]['relevance_score'].values for name in configs]
bp = axes[1].boxplot(relevance_data, labels=list(configs.keys()), patch_artist=True,
                      widths=0.6, showmeans=True, meanprops={"marker": "D", "markerfacecolor": "red"})

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.2)

axes[1].set_ylabel('Relevance Score (1-5)', fontsize=12)
axes[1].set_title('Relevance Score Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 5.8)

# 添加均值标记
for i, (mean, color) in enumerate(zip(relevance_means, colors)):
    axes[1].scatter(i + 1, mean, marker='D', color='red', s=50, zorder=5)

plt.tight_layout()
plt.savefig(output_dir / 'relevance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  [完成] relevance_comparison.png")

# ============== 图2: Task Completion Rate 对比 ==============
fig, ax = plt.subplots(figsize=(10, 6))

completion_rates = [dfs[name]['task_completion'].mean() * 100 for name in configs]
bars = ax.bar(list(configs.keys()), completion_rates, color=colors, edgecolor='black',
              linewidth=1.2, alpha=0.85)

# 添加数值标签
for bar, rate in zip(bars, completion_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 2,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Task Completion Rate (%)', fontsize=12)
ax.set_xlabel('Configuration', fontsize=12)
ax.set_title('Task Completion Rate by Configuration', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(output_dir / 'completion_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  [完成] completion_comparison.png")

# ============== 图3: Latency 对比 ==============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 延迟分解（对数刻度）
retrieval_latencies = [dfs[name]['retrieval_latency'].mean() * 1000 for name in configs]  # 转为ms
generation_latencies = [dfs[name]['generation_latency'].mean() for name in configs]  # 秒

x = np.arange(len(configs))
width = 0.35

# 检索延迟
bars1 = axes[0].bar(x - width/2, retrieval_latencies, width, label='Retrieval Latency',
                    color='#9b59b6', edgecolor='black', alpha=0.85)

# 生成延迟
bars2 = axes[0].bar(x + width/2, generation_latencies, width, label='Generation Latency',
                     color='#e74c3c', edgecolor='black', alpha=0.85)

axes[0].set_ylabel('Latency', fontsize=12)
axes[0].set_title('Latency Breakdown', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(list(configs.keys()), rotation=15, ha='right')
axes[0].legend(loc='upper left')

# 设置对数刻度
axes[0].set_yscale('log')
axes[0].set_ylabel('Latency (ms for retrieval, s for generation)', fontsize=11)

# 添加数值标注
for bar in bars1:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, height * 1.1,
                  f'{height:.1f}ms', ha='center', va='bottom', fontsize=8, rotation=0)

for bar in bars2:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, height * 1.1,
                  f'{height:.1f}s', ha='center', va='bottom', fontsize=8, rotation=0)

# 总延迟对比
total_latencies = [dfs[name]['total_latency'].mean() for name in configs]
bars = axes[1].bar(list(configs.keys()), total_latencies, color=colors, edgecolor='black',
                    linewidth=1.2, alpha=0.85)

for bar, latency in zip(bars, total_latencies):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                  f'{latency:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[1].set_ylabel('Total Latency (s)', fontsize=12)
axes[1].set_xlabel('Configuration', fontsize=12)
axes[1].set_title('Average Total Latency', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'latency_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  [完成] latency_comparison.png")

# ============== 图4: 回答长度对比 ==============
fig, ax = plt.subplots(figsize=(10, 6))

word_lengths = [dfs[name]['response_length_words'].mean() for name in configs]
bars = ax.bar(list(configs.keys()), word_lengths, color=colors, edgecolor='black',
              linewidth=1.2, alpha=0.85)

for bar, length in zip(bars, word_lengths):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f'{length:.0f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('Average Response Length (words)', fontsize=12)
ax.set_xlabel('Configuration', fontsize=12)
ax.set_title('Average Response Length by Configuration', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'response_length_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  [完成] response_length_comparison.png")

# ============== 图5: 综合对比雷达图 ==============
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

categories = ['Relevance\n(norm)', 'Completion\nRate', 'Speed\n(inverted)',
              'Conciseness', 'Retrieval\nQuality']

# 计算归一化数据
max_relevance = 5.0
max_latency = max(total_latencies)
max_words = max(word_lengths)

def _clip01(x: float) -> float:
    """Clamp radar axis values to [0, 100] for consistent scale."""
    return float(np.clip(x, 0.0, 100.0))


data_radar = {}
for name in configs:
    df = dfs[name]
    # Conciseness: shorter answers → higher score, vs global max length only (no +offset)
    conciseness = (1 - df["response_length_words"].mean() / max_words) * 100
    data_radar[name] = [
        _clip01(df["relevance_score"].mean() / max_relevance * 100),  # Relevance
        _clip01(df["task_completion"].mean() * 100),  # Completion rate
        _clip01((1 - df["total_latency"].mean() / max_latency) * 100),  # Speed (inverted)
        _clip01(conciseness),  # Conciseness (was +20 before, caused >100 for very short outputs)
        70,  # 占位：检索质量未单独建模，保持与旧版一致
    ]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

markers = ['o', 's', '^', 'D']
for i, (name, values) in enumerate(data_radar.items()):
    values += values[:1]
    ax.plot(angles, values, f'{markers[i]}-', linewidth=2.5, label=name, markersize=8)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=10)
ax.set_title('Multi-Dimensional Comparison (Radar Chart)', fontsize=14, fontweight='bold', pad=25)

plt.tight_layout()
plt.savefig(output_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  [完成] radar_comparison.png")

# ============== 图6: 每问题详细对比热力图 ==============
fig, ax = plt.subplots(figsize=(12, 8))

# 准备数据
query_ids = [f"Q{i:02d}" for i in range(1, 21)]
heatmap_data = []

for name in configs:
    heatmap_data.append(dfs[name]['relevance_score'].values)

heatmap_data = np.array(heatmap_data)

# 创建热力图
im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)

ax.set_xticks(range(len(query_ids)))
ax.set_xticklabels(query_ids, fontsize=9)
ax.set_yticks(range(len(configs)))
ax.set_yticklabels(list(configs.keys()), fontsize=11, fontweight='bold')

ax.set_xlabel('Query ID', fontsize=12)
ax.set_ylabel('Configuration', fontsize=12)
ax.set_title('Relevance Score Heatmap (All Queries)', fontsize=14, fontweight='bold')

# 添加数值标注
for i in range(len(configs)):
    for j in range(len(query_ids)):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.0f}',
                       ha='center', va='center', color='black', fontsize=8, fontweight='bold')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Relevance Score', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'relevance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  [完成] relevance_heatmap.png")

# ============== 生成汇总表格 ==============
print("\n生成汇总统计表...")

summary_data = []
for name in configs:
    df = dfs[name]
    summary_data.append({
        'Configuration': name,
        'Avg Relevance (1-5)': round(df['relevance_score'].mean(), 2),
        'Relevance Std': round(df['relevance_score'].std(), 2),
        'Min Relevance': df['relevance_score'].min(),
        'Max Relevance': df['relevance_score'].max(),
        'Task Completion Rate': f"{df['task_completion'].mean()*100:.1f}%",
        'Avg Words': round(df['response_length_words'].mean(), 1),
        'Avg Retrieval Latency (ms)': round(df['retrieval_latency'].mean()*1000, 2),
        'Avg Generation Latency (s)': round(df['generation_latency'].mean(), 3),
        'Avg Total Latency (s)': round(df['total_latency'].mean(), 3),
    })

summary_df = pd.DataFrame(summary_data)
summary_path = output_dir / 'comparison_summary.csv'
summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
print(f"  [完成] comparison_summary.csv")

# ============== 生成详细报告 ==============
report_path = output_dir / 'analysis_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("RAG System Evaluation Analysis Report\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. CONFIGURATION SUMMARY\n")
    f.write("-" * 40 + "\n")
    for name in configs:
        f.write(f"  {name}: {len(dfs[name])} queries evaluated\n")

    f.write("\n2. KEY METRICS\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'Config':<20} {'Relevance':<12} {'Completion':<15} {'Latency':<12}\n")
    for i, name in enumerate(configs):
        df = dfs[name]
        rel = f"{df['relevance_score'].mean():.2f} +/- {df['relevance_score'].std():.2f}"
        comp = f"{df['task_completion'].mean()*100:.1f}%"
        lat = f"{df['total_latency'].mean():.2f}s"
        f.write(f"{name:<20} {rel:<12} {comp:<15} {lat:<12}\n")

    f.write("\n3. KEY FINDINGS\n")
    f.write("-" * 40 + "\n")

    # 找出最佳配置
    best_relevance = max(configs.keys(), key=lambda x: dfs[x]['relevance_score'].mean())
    best_completion = max(configs.keys(), key=lambda x: dfs[x]['task_completion'].mean())
    fastest = min(configs.keys(), key=lambda x: dfs[x]['total_latency'].mean())

    f.write(f"  - Best Relevance Score: {best_relevance}\n")
    f.write(f"    ({dfs[best_relevance]['relevance_score'].mean():.2f}/5.0)\n\n")
    f.write(f"  - Best Task Completion: {best_completion}\n")
    f.write(f"    ({dfs[best_completion]['task_completion'].mean()*100:.1f}%)\n\n")
    f.write(f"  - Fastest Response: {fastest}\n")
    f.write(f"    ({dfs[fastest]['total_latency'].mean():.2f}s avg)\n\n")

    f.write("4. LLM COMPARISON\n")
    f.write("-" * 40 + "\n")
    for llm in ['Flan-T5', 'Mistral-7B']:
        llm_configs = [c for c in configs.keys() if llm in c]
        if llm_configs:
            avg_rel = np.mean([dfs[c]['relevance_score'].mean() for c in llm_configs])
            avg_comp = np.mean([dfs[c]['task_completion'].mean() for c in llm_configs])
            avg_lat = np.mean([dfs[c]['total_latency'].mean() for c in llm_configs])
            f.write(f"  {llm}:\n")
            f.write(f"    Avg Relevance: {avg_rel:.2f}/5.0\n")
            f.write(f"    Avg Completion: {avg_comp*100:.1f}%\n")
            f.write(f"    Avg Latency: {avg_lat:.2f}s\n\n")

    f.write("5. CHUNKING COMPARISON\n")
    f.write("-" * 40 + "\n")
    for chunk in ['256', '512']:
        chunk_configs = [c for c in configs.keys() if chunk in c]
        if chunk_configs:
            avg_rel = np.mean([dfs[c]['relevance_score'].mean() for c in chunk_configs])
            avg_comp = np.mean([dfs[c]['task_completion'].mean() for c in chunk_configs])
            f.write(f"  Chunk-{chunk}:\n")
            f.write(f"    Avg Relevance: {avg_rel:.2f}/5.0\n")
            f.write(f"    Avg Completion: {avg_comp*100:.1f}%\n\n")

print(f"  [完成] analysis_report.txt")

# ============== 打印汇总表 ==============
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print(f"所有文件已保存至: {output_dir}")
print("=" * 70)
print("\n生成的文件列表:")
for f in sorted(output_dir.glob('*')):
    print(f"  - {f.name}")
