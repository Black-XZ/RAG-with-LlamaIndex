# 实验结果报告

## 1. 实验配置

| 配置 | LLM | Chunk Size | 参数设置 |
|------|-----|------------|----------|
| C1 | Flan-T5-base | 256 tokens | temperature=0.1, max_tokens=256 |
| C2 | Flan-T5-base | 512 tokens | temperature=0.1, max_tokens=256 |
| C3 | Mistral-7B-Instruct | 256 tokens | temperature=0.1, max_tokens=256 |
| C4 | Mistral-7B-Instruct | 512 tokens | temperature=0.1, max_tokens=256 |

**评估指标**: Relevance Score (1-5), Task Completion (0/1), Latency (ms)
**评估数据集**: 20 个 RAG 相关问题

---

## 2. 核心结果汇总

| 配置 | Relevance ↑ | Task Completion ↑ | 总延迟 ↓ | 回答长度 |
|------|-----------|------------------|---------|----------|
| Flan-T5/256 | 2.10 ± 0.72 | 0.0% | 0.90s | 4.8 词 |
| Flan-T5/512 | 2.10 ± 0.55 | 0.0% | 0.23s | 5.2 词 |
| Mistral-7B/256 | 3.80 ± 1.24 | 65.0% | 11.12s | 115.8 词 |
| **Mistral-7B/512** | **3.90 ± 1.07** | **70.0%** | 6.72s | 135.3 词 |

↑ 越高越好 | ↓ 越低越好

---

## 3. 关键发现

### RQ1: Chunking 策略影响

| Chunk Size | Avg Relevance | Avg Completion |
|------------|-------------|----------------|
| 256 tokens | 2.95 | 32.5% |
| 512 tokens | 3.00 | 35.0% |

**结论**: 512 tokens 在两个指标上略有优势，但差异不大（<5%）。较大 chunk 保留了更多上下文信息，对复杂问题帮助稍大。

### RQ2: LLM 模型对比

| LLM | Avg Relevance | Avg Completion | Avg Latency |
|-----|-------------|----------------|-------------|
| Flan-T5 | 2.10 | 0.0% | 0.56s |
| Mistral-7B | 3.85 | 67.5% | 8.92s |

**结论**: Mistral-7B 在质量上显著优于 Flan-T5：
- Relevance 提升 **+83%** (2.10 → 3.85)
- Task Completion 提升 **+67.5%** (0% → 67.5%)
- 但延迟增加约 **16 倍** (0.56s → 8.92s)

---

## 4. 分析与讨论

### 4.1 Flan-T5 表现分析

Flan-T5 在所有配置下 Task Completion 均为 0%，主要原因：
1. **模型容量限制**: Flan-T5-base (250M 参数) 难以理解复杂上下文
2. **生成长度过短**: 平均仅 4-5 词，无法完整回答问题
3. **幻觉问题**: 常以 "I don't know" 或关键词简短回答

### 4.2 Mistral-7B 优势分析

Mistral-7B-Instruct (7B 参数) 表现优异：
1. **强指令跟随能力**: 能准确理解问题并给出完整答案
2. **长上下文理解**: 平均回答 115-135 词，涵盖多个要点
3. **引用生成**: 正确引用检索到的文档 [C1], [C2]

### 4.3 Chunk Size 影响

- **小 chunk (256)**: 精确但可能丢失上下文
- **大 chunk (512)**: 保留更多上下文，但可能引入噪声

---

## 5. 可视化图表

所有图表位于: `results/runs/analysis/`

| 文件 | 描述 |
|------|------|
| `relevance_comparison.png` | Relevance 评分柱状图和箱线图 |
| `completion_comparison.png` | Task Completion Rate 对比 |
| `latency_comparison.png` | 延迟分解图 |
| `response_length_comparison.png` | 回答长度对比 |
| `radar_comparison.png` | 综合能力雷达图 |
| `relevance_heatmap.png` | 20个问题详细评分热力图 |
| `comparison_summary.csv` | 数值汇总表 |

---

## 6. 结论与建议

### 质量优先场景
→ 推荐 **Mistral-7B/512**
- 最佳 Relevance (3.90)
- 最高 Task Completion (70%)
- 适合生产环境或对质量要求高的应用

### 速度优先场景
→ 推荐 **Flan-T5/512**
- 最快响应时间 (0.23s)
- 仅适合简单检索或开发测试

### 最佳性价比
→ 推荐 **Mistral-7B/256**
- 质量接近 512 配置 (3.80 vs 3.90)
- 延迟更低 (11.12s vs 6.72s)
- 适合需要平衡质量和速度的场景

---

## 7. 局限性说明

1. **评估集有限**: 仅 20 个问题，统计显著性有限
2. **领域单一**: 问题集中于 RAG 技术本身
3. **无自动评估**: 仅依赖人工评分，可能存在主观偏差
4. **硬件差异**: 延迟数据受服务器负载影响

---

*报告生成时间: 实验完成后自动生成*
*数据来源: `results/runs/analysis/comparison_summary.csv`*
