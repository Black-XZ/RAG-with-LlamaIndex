# Progress Report - Building and Evaluating a Document QA System with LlamaIndex

**Project:** Building and Evaluating a Document Question Answering System
**Framework:** LlamaIndex
**Date:** 2026-03-21
**Author:** 项目团队

---

## 1. Overview

### 1.1 Project Summary

本项目实现了一个基于检索增强生成（RAG）架构的文档问答（DocQA）系统。系统结合向量检索与大型语言模型生成，基于文档语料库回答用户查询。实验评估了两种分块策略（256/512 tokens）与两种 LLM 后端（Mistral-7B-Instruct、Flan-T5-base）在 20 个测试问题上的表现，涵盖 Relevance、Task Completion、延迟等多个维度。

### 1.2 Objectives

| Objective | Status | Notes |
|-----------|--------|-------|
| 实现文档加载管道 | ✅ 完成 | 支持 PDF 和 TXT 格式 |
| 构建可配置分块系统 | ✅ 完成 | 256 和 512 token 策略，10% overlap |
| 创建向量索引 | ✅ 完成 | 使用 Sentence-BERT (all-MiniLM-L6-v2) |
| 集成 Mistral-7B-Instruct | ✅ 完成 | 支持量化加速 |
| 集成 Flan-T5-base | ✅ 完成 | 轻量级基线模型 |
| 开发评估框架 | ✅ 完成 | 人工 + 自动指标 |
| 运行对比实验 | ✅ 完成 | 4 组配置全部完成 |

### 1.3 Research Questions

- **RQ1:** 分块策略（256 vs 512 tokens，10% overlap）如何影响 RAG 质量与效率？
- **RQ2:** 在相同检索条件下，Mistral-7B 与 Flan-T5 在响应质量、任务完成率和延迟方面的表现如何？

---

## 2. System Design

### 2.1 Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐     ┌─────────────┐     ┌────────────────┐    │
│  │ Documents │────>│   Loader    │────>│   Chunker      │    │
│  │ (PDF/TXT)│     │  & Cleaning │     │ (256/512 tok)  │    │
│  └──────────┘     └─────────────┘     └───────┬────────┘    │
│                                               │               │
│                                               v               │
│                                       ┌─────────────┐        │
│                                       │  Embedding  │        │
│                                       │  (SBERT)    │        │
│                                       └──────┬──────┘        │
│                                              │               │
│                                              v               │
│  ┌──────────┐     ┌─────────────┐     ┌─────────────┐      │
│  │ Response │<────│     LLM     │<────│   Retriever │      │
│  │ +Citations│    │ (Mistral/  │     │   (Top-K)   │      │
│  └──────────┘     │  Flan-T5)   │     └─────────────┘      │
│                   └─────────────┘                           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 Module Implementation Status

| Module | File | Status | Description |
|--------|------|--------|-------------|
| Document Loader | `src/loaders.py` | ✅ | PDF/TXT 加载与清洗 |
| Index Builder | `src/indexing.py` | ✅ | 可配置分块 + FAISS 向量索引 |
| Retriever | `src/retriever.py` | ✅ | 向量相似度检索 (Top-K) |
| LLM Backends | `src/llm_backends.py` | ✅ | Mistral + Flan-T5 统一接口 |
| RAG Pipeline | `src/rag_pipeline.py` | ✅ | 端到端编排 |
| Evaluation | `src/eval_protocol.py` | ✅ | 指标定义 + 人工评分模板 |
| Metrics | `src/metrics.py` | ✅ | 指标计算工具 |
| Utils | `src/utils.py` | ✅ | 通用工具函数 |

### 2.3 Key Configuration Parameters

```yaml
# 分块策略 S1 (细粒度)
chunk_size: 256
overlap_ratio: 0.1

# 分块策略 S2 (粗粒度)
chunk_size: 512
overlap_ratio: 0.1

# Embedding
model: sentence-transformers/all-MiniLM-L6-v2
dimension: 384

# LLM Settings
Mistral-7B-Instruct:
  max_tokens: 256
  temperature: 0.1
  quantization: 4-bit (optional)

Flan-T5-base:
  max_tokens: 256
  temperature: 0.1
```

---

## 3. Experiment Plan

### 3.1 Experimental Configurations

| Config | LLM | Chunk | 描述 |
|--------|-----|-------|------|
| C1 | Mistral-7B-Instruct | 256 | 强模型 + 细粒度分块 |
| C2 | Mistral-7B-Instruct | 512 | 强模型 + 粗粒度分块 |
| C3 | Flan-T5-base | 256 | 轻量模型 + 细粒度分块 |
| C4 | Flan-T5-base | 512 | 轻量模型 + 粗粒度分块 |

### 3.2 Evaluation Metrics

| Metric | Type | Scale | Collection |
|--------|------|-------|------------|
| Response Relevance | Human | 1-5 | 人工标注 |
| Task Completion | Binary | 0/1 | 人工/混合 |
| Response Length | Automated | Words | 系统 |
| Retrieval Latency | Automated | ms | 系统 |
| Generation Latency | Automated | s | 系统 |

### 3.3 Query Set

- **规模:** 20 个查询
- **主题:** RAG 基础、分块策略、检索方法、LLM 对比、评估指标
- **格式:** JSONL（含查询 ID、问题、评分）

---

## 4. Current Progress

### 4.1 Completed Tasks ✅

- [x] 项目结构创建
- [x] 配置文件（chunking_256.yaml, chunking_512.yaml）
- [x] 样本文档语料库（6 篇技术文档）
- [x] 评估查询集（20 个查询）
- [x] 所有源代码模块（loaders, indexing, retriever, llm_backends, rag_pipeline, eval_protocol, metrics, utils）
- [x] CLI 脚本（run_build_index.py, run_eval.py）
- [x] Jupyter notebook 演示
- [x] 报告大纲
- [x] **进度报告模板（已填入实验数据）**
- [x] **实验结果报告（REPORT_RESULTS.md）**
- [x] **可视化图表生成（results/runs/analysis/）**

### 4.2 In Progress 🔄

| Task | Progress | Notes |
|------|----------|-------|
| Mistral-7B Integration | 100% | 支持量化加速 |
| Index Building | 100% | 256/512 分块均可用 |
| Evaluation Framework | 100% | 人工+自动指标 |
| Visualization Generation | 100% | 雷达图、柱状图、热力图等 |

### 4.3 Pending Tasks ⏳

- [ ] 撰写 Discussion 章节
- [ ] 生成最终综合报告

### 4.4 Current Experiment Status

```
Configuration         Status    Relevance    Completion    Latency
───────────────────────────────────────────────────────────────────
C1 (M7B/256)         ✅ Done    3.80         65.0%         11.12s
C2 (M7B/512)         ✅ Done    3.90         70.0%          6.72s
C3 (T5/256)          ✅ Done    2.10          0.0%          0.90s
C4 (T5/512)          ✅ Done    2.10          0.0%          0.23s
```

---

## 5. Risks and Mitigations

### 5.1 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU 显存不足以运行 Mistral-7B | Medium | High | 实现 4-bit 量化降级 |
| PDF 解析质量问题 | Low | Medium | 混合提取 + 手动修正 |
| 人工评估时间约束 | Low | Medium | 开发半自动评估工具 |
| 语料库规模限制 | Medium | Low | 聚焦技术准确性指标 |

### 5.2 Mitigation Strategies

1. **GPU 资源约束:** 优先 Flan-T5 实验；Mistral 需要时启用 4-bit 量化
2. **人工评估工作量:** 制定清晰标注指南；每次评估 10-15 分钟
3. **PDF 处理质量:** 添加验证检查；必要时手动替换文档
4. **时间线:** 利用不同 LLM 后端并行实验；批量评估查询

### 5.3 Contingency Plans

| Scenario | Response |
|----------|----------|
| Mistral-7B 加载失败 | 使用 Flan-T5 作为主要模型；报告模型可用性挑战 |
| 评估数据有限 | 聚焦自动指标；注明人工评估局限性 |
| 检索质量差 | 分析失败案例；相应调整分块策略 |

---

## 6. TODO Checklist

### Immediate (已完成)

- [x] 验证两种分块配置的索引构建
- [x] 完成 Flan-T5 基线实验
- [x] 配置 Mistral-7B（含量化）
- [x] 创建人工评估模板
- [x] 运行全部 4 组配置实验
- [x] 收集人工评估数据
- [x] 生成对比可视化图表
- [x] 完成结果分析

### Short-term

- [x] 完成全部 4 组配置实验
- [x] 收集人工评估数据
- [x] 生成对比图表
- [x] 起草结果分析章节
- [ ] 撰写 Discussion 章节
- [ ] 生成最终综合报告

### Medium-term

- [ ] 完善最终报告
- [ ] 准备展示材料
- [ ] 提交最终交付物

---

## 7. Appendix

### A. File Structure

```
project-root/
├── data/
│   ├── corpus/          # Documents (PDF/TXT)
│   └── queries.jsonl    # Evaluation queries
├── configs/
│   ├── chunking_256.yaml
│   └── chunking_512.yaml
├── src/
│   ├── loaders.py
│   ├── indexing.py
│   ├── retriever.py
│   ├── llm_backends.py
│   ├── rag_pipeline.py
│   ├── eval_protocol.py
│   ├── metrics.py
│   ├── utils.py
│   ├── run_build_index.py
│   └── run_eval.py
├── results/
│   └── runs/
│       ├── eval_flant5_256/
│       ├── eval_flant5_512/
│       ├── eval_mistral7b_256/
│       ├── eval_mistral7b_512/
│       └── analysis/
│           ├── radar_comparison.png
│           ├── relevance_comparison.png
│           ├── completion_comparison.png
│           ├── latency_comparison.png
│           ├── response_length_comparison.png
│           ├── relevance_heatmap.png
│           ├── comparison_summary.csv
│           └── analysis_report.txt
├── notebooks/
│   └── demo_end2end.ipynb
├── REPORT_RESULTS.md
├── REPORT_OUTLINE.md
├── PROGRESS_REPORT_TEMPLATE.md
├── README.md
└── requirements.txt
```

### B. Reproduction Commands

```bash
# 安装依赖
pip install -r requirements.txt

# 构建索引
python src/run_build_index.py --config configs/chunking_256.yaml
python src/run_build_index.py --config configs/chunking_512.yaml

# 运行评估
python src/run_eval.py --llm flant5 --chunk 256
python src/run_eval.py --llm flant5 --chunk 512
python src/run_eval.py --llm mistral7b --chunk 256 --use-quantization
python src/run_eval.py --llm mistral7b --chunk 512 --use-quantization

# 生成可视化
python src/generate_visualizations.py

# 查看结果
# results/runs/analysis/ 下包含所有图表和报告
```

### C. 实验数据汇总

| 配置 | Relevance (1-5) | 任务完成率 | 延迟 (s) | 回答词数 |
|------|----------------|-----------|----------|---------|
| Flan-T5/256 | 2.10 ± 0.72 | 0.0% | 0.90 | 4.8 |
| Flan-T5/512 | 2.10 ± 0.55 | 0.0% | 0.23 | 5.2 |
| Mistral-7B/256 | 3.80 ± 1.24 | 65.0% | 11.12 | 115.8 |
| Mistral-7B/512 | **3.90 ± 1.07** | **70.0%** | 6.72 | 135.3 |

### D. 可视化输出

所有图表位于 `results/runs/analysis/`：

| 文件 | 描述 |
|------|------|
| `radar_comparison.png` | 综合能力雷达图 |
| `relevance_comparison.png` | Relevance 评分柱状图和箱线图 |
| `completion_comparison.png` | Task Completion Rate 对比 |
| `latency_comparison.png` | 延迟分解图 |
| `response_length_comparison.png` | 回答长度对比 |
| `relevance_heatmap.png` | 20 个问题详细评分热力图 |
| `comparison_summary.csv` | 数值汇总表 |
| `analysis_report.txt` | 文本分析报告 |

---

*Progress Report - Final Version (2026-03-21)*
