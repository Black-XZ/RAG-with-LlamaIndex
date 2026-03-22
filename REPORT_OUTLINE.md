# Building and Evaluating a Document Question Answering System with LlamaIndex: A Study of Chunking Strategies and LLM Backends

## Project Report Outline

---

## 1. Introduction

### 1.1 Background and Motivation

近年来，大型语言模型（LLMs）在自然语言理解和生成方面展现出卓越的能力。然而，其固有局限性——包括知识截止日期和幻觉倾向——催生了检索增强生成（RAG）技术的发展。RAG 将神经检索与 LLM 生成相结合，能够产生更准确、更及时、更可验证的响应。

文档问答（Document QA / DocQA）是 RAG 的一个重要应用场景，系统基于文档语料库回答用户查询。这一方法在企业知识库、学术文献综述和专业领域应用中对准确性和来源追溯尤为关键。有效评估 RAG 系统在不同配置下的性能，对于系统设计和部署具有重要意义。

### 1.2 Research Questions

本研究聚焦以下两个核心研究问题：

- **RQ1**: 分块策略（256 vs 512 tokens，10% overlap）如何影响基于 RAG 的文档问答系统的质量与效率？

- **RQ2**: 在相同检索条件下，不同 LLM 后端（Mistral-7B-Instruct vs Flan-T5-base）在响应质量、任务完成率和延迟方面表现如何？

### 1.3 Contributions

本研究的主要贡献包括：
1. **系统性评估**: 在受控检索条件下，系统评估了分块策略对 RAG 性能的影响
2. **LLM 对比分析**: 在相同检索上下文下，对 Mistral-7B-Instruct 和 Flan-T5-base 进行了全面对比
3. **可复现框架**: 基于 LlamaIndex 构建了完整的实验框架，涵盖数据加载、索引构建、检索生成和评估全流程
4. **综合评估协议**: 提出了结合 Relevance（1-5）、Task Completion（二元）、延迟和回答长度的多维评估指标体系

---

## 2. System Architecture

### 2.1 Overview

RAG 系统架构由四个核心组件构成：文档处理（Document Processing）、索引构建（Index Construction）、检索（Retrieval）和生成（Generation）。下图展示了完整的处理流水线：

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG System Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌───────────┐    ┌─────────────┐               │
│  │ Documents │───>│  Loader   │───>│  Chunker    │               │
│  │ (PDF/TXT)│    │  & Clean   │    │  (256/512)  │               │
│  └──────────┘    └───────────┘    └──────┬──────┘               │
│                                            │                      │
│                                            v                      │
│  ┌───────────┐    ┌────────────┐    ┌─────────────┐             │
│  │  Query    │───>│  Retrieval │<───│  Embedding  │             │
│  │  Input    │    │  (Top-K)   │    │  (SBERT)    │             │
│  └───────────┘    └─────┬──────┘    └─────────────┘             │
│                          │                                       │
│                          v                                       │
│  ┌───────────┐    ┌────────────┐    ┌─────────────┐             │
│  │ Response  │<───│    LLM     │<───│   Prompt    │             │
│  │ +Citation │    │ Generation │    │  Assembly   │             │
│  └───────────┘    └────────────┘    └─────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Descriptions

#### 2.2.1 Document Loader (`src/loaders.py`)

文档加载模块负责 PDF 和 TXT 文档的加载与预处理。核心功能包括：自动文本提取、页眉页脚清洗、空白字符规范化，以及转换为 LlamaIndex Document 对象。模块支持多种文本编码，并自动跳过损坏或空的文档。

#### 2.2.2 Index Builder (`src/indexing.py`)

索引构建模块实现了可配置的分块策略和向量索引构建。支持基于 token 的分块（256/512 tokens）和可配置的 overlap ratio（10%），使用 Sentence-BERT (all-MiniLM-L6-v2, 384 维) 生成 chunk embeddings，并基于 FAISS 构建向量索引以支持高效相似度检索。

#### 2.2.3 Retriever (`src/retriever.py`)

检索组件使用 FAISS 向量相似度搜索，为给定查询返回 top-k（k=2）最相关的文档 chunk。检索结果包含 chunk 文本、节点 ID 和相似度分数，供后续生成阶段使用。

#### 2.2.4 LLM Backends (`src/llm_backends.py`)

实现了两个 LLM 后端：(1) **Mistral-7B-Instruct** 用于高质量生成，支持可选 4-bit 量化以降低显存需求；(2) **Flan-T5-base** 作为轻量级基线。两者共享统一接口，便于实验对比。

#### 2.2.5 RAG Pipeline (`src/rag_pipeline.py`)

RAG 流水线编排检索与生成流程，包括：检索上下文组装、Prompt 构建（含 [C1], [C2] 等引用标记）、LLM 生成、以及响应格式化输出。

---

## 3. Methodology

### 3.1 Data

本研究使用一个包含 6 篇技术文档的语料库，内容涵盖 RAG 概念、分块策略、检索方法和 LLM 评估。每个文档包含约 500-800 词的技术内容。

### 3.2 Chunking Strategies

| Strategy | Chunk Size | Overlap | Tokens | Description |
|----------|------------|---------|--------|-------------|
| S1 | 256 | 10% | 25 | Fine-grained, high precision |
| S2 | 512 | 10% | 51 | Broader context, balanced |

### 3.3 LLM Configurations

| Model | Parameters | Quantization | Max Tokens | Temperature |
|-------|------------|--------------|------------|-------------|
| Mistral-7B-Instruct | 7B | 4-bit (optional) | 256 | 0.1 |
| Flan-T5-base | 220M | None | 256 | 0.1 |

### 3.4 Evaluation Protocol

#### 3.4.1 Metrics

1. **Response Relevance (1-5)**: 人工评估答案质量与相关性，采用 5 分李克特量表
2. **Task Completion (0/1)**: 二元判断，问题是否被正确完整回答
3. **Response Length**: 字数和词数统计
4. **Latency**: 检索延迟（ms）、生成延迟（s）和总延迟（s）

#### 3.4.2 Query Set

20 个评估查询，覆盖 RAG 系统的各个维度：基础概念、分块策略、检索机制、模型对比、评估指标等。查询以 JSONL 格式存储，包含 query_id、question 和预期回答类型。

### 3.5 Experimental Configurations

| Configuration | LLM | Chunk Size | Description |
|---------------|-----|------------|-------------|
| C1 | Mistral-7B | 256 | Strong model + fine chunks |
| C2 | Mistral-7B | 512 | Strong model + coarse chunks |
| C3 | Flan-T5 | 256 | Light model + fine chunks |
| C4 | Flan-T5 | 512 | Light model + coarse chunks |

---

## 4. Experiments and Results

### 4.1 Experimental Setup

[PLACEHOLDER: All experiments were conducted on [hardware configuration]. Each configuration was run across all 20 evaluation queries with consistent random seeds for reproducibility.]

### 4.2 Results

表 1 展示了四种配置的核心指标对比。

**表 1: RAG 系统配置对比**

| Configuration | Avg Relevance (1-5) | Relevance Std | Task Completion Rate | Avg Response Length (words) | Avg Total Latency (s) |
|---------------|---------------------|---------------|----------------------|----------------------------|-----------------------|
| Flan-T5 / 256 | 2.10 | ±0.72 | 0.0% | 4.8 | 0.90 |
| Flan-T5 / 512 | 2.10 | ±0.55 | 0.0% | 5.2 | 0.23 |
| Mistral-7B / 256 | 3.80 | ±1.24 | 65.0% | 115.8 | 11.12 |
| **Mistral-7B / 512** | **3.90** | **±1.07** | **70.0%** | 135.3 | 6.72 |

**↑ 越高越好 | ↓ 越低越好**

### 4.3 Key Findings

#### RQ1: Effect of Chunking Strategy

| Chunk Size | Avg Relevance (1-5) | Avg Completion Rate | Avg Latency (s) |
|------------|---------------------|----------------------|------------------|
| 256 tokens | 2.95 | 32.5% | 6.01 |
| 512 tokens | 3.00 | 35.0% | 3.47 |

**结论:** 512 tokens 在 Relevance 和 Completion 上略有优势（约 +2%），延迟显著更低（3.47s vs 6.01s），因为 Mistral-7B 在 512 配置下的生成步数更少（6.72s vs 11.12s）。较大 chunk 保留了更多上下文信息，对复杂问题的理解帮助更大；但差异相对有限（<5%），说明在当前语料规模下，分块策略的影响不如 LLM 选择显著。

#### RQ2: Effect of LLM Backend

| LLM | Avg Relevance (1-5) | Avg Completion Rate | Avg Latency (s) | Avg Response Words |
|-----|---------------------|----------------------|------------------|--------------------|
| Flan-T5 | 2.10 | 0.0% | 0.56 | 5.0 |
| Mistral-7B | 3.85 | 67.5% | 8.92 | 125.6 |

**结论:** Mistral-7B 在质量上显著优于 Flan-T5：Relevance 提升 **+83%**（2.10 → 3.85），Task Completion 从 0% 跃升至 67.5%，但延迟增加约 **16 倍**（0.56s → 8.92s）。Flan-T5 生成极短回答（平均仅 5 词），无法完成任何问答任务，但 Flan-T5/512 的延迟仅 0.23s，适合对速度有极端要求的简单场景。

**最佳性价比:** Mistral-7B/512 在 Relevance（3.90）、Completion（70%）和延迟（6.72s）之间取得最佳平衡。

---

## 5. Discussion

### 5.1 Memory-Performance Trade-off

实验揭示了 RAG 系统中一个核心权衡：**分块粒度与检索精度的平衡**。

- **256 tokens (细粒度)**: 精确匹配度高，适合简单事实型问题（"What is RAG?"），但可能丢失跨段落语义关联。当答案分布在多个 chunk 时，top-k 检索容易遗漏关键信息。
- **512 tokens (粗粒度)**: 保留更多段落级上下文，减少信息断裂，但可能引入更多噪声，降低检索 precision。实验中 512 配置在综合质量上略优，说明当前语料的段落语义完整性比精确性更重要。

这一权衡在不同领域可能表现不同：医学/法律等需要精确引用的领域可能偏好细粒度，而开放域知识问答则可受益于粗粒度的上下文丰富度。

### 5.2 LLM Capability Differences

两种 LLM 在相同检索上下文下表现差异巨大，根本原因在于：

1. **指令跟随能力**: Flan-T5-base 是纯预训练模型，缺乏指令微调（Instruction Tuning），难以将检索到的上下文转化为流畅的问答回答。Mistral-7B-Instruct 经过指令微调，能更好理解用户问题意图。
2. **上下文窗口利用率**: Flan-T5 平均生成仅 5 词，远低于 `max_tokens=256` 的上限，说明模型无法有效利用检索到的丰富上下文。Mistral-7B 平均生成 115-135 词，充分利用了上下文信息。
3. **幻觉行为**: Flan-T5 常以 "I don't know" 或单个关键词回答，即便检索到了相关 chunk 也无法正确整合。Mistral-7B 能引用具体文档 [C1], [C2] 并综合多段信息。

**质量-延迟权衡**: Mistral-7B 的质量优势（Relevance +83%, Completion +67.5%）是以约 16 倍延迟为代价的。在实时性要求高的场景（如客服），可考虑 Flan-T5/512 作为快速过滤层，再对高置信度问题路由至 Mistral-7B。

### 5.3 Failure Cases and Error Analysis

通过对 20 个问题的逐题分析，识别出以下失败模式：

| Failure Mode | 主要影响配置 | 描述 |
|-------------|------------|------|
| **检索为空/低相关** | 全部配置 | 部分问题未检索到相关 chunk，导致回答不完整 |
| **生成幻觉** | Flan-T5 为主 | 模型生成了与检索上下文不符的信息 |
| **回答过短/无意义** | Flan-T5 | 回答长度 < 5 词，无法传达有效信息 |
| **延迟过高** | Mistral-7B | 生成延迟 6-11s，影响用户体验 |
| **引用缺失** | Flan-T5 | 无法提供文档引用，降低答案可信度 |

Mistral-7B 的失败主要集中在需要精确数值或多跳推理的问题上（如"哪些文档提到向量索引？"），而 Flan-T5 的失败是系统性的——几乎所有问题都无法给出有效回答。

### 5.4 Limitations

本研究存在以下局限性：

1. **评估集规模**: 仅 20 个查询，统计功效有限。每个配置约 4% 的完成率差异在统计上可能不显著。建议未来工作使用至少 100+ 查询的评估集。
2. **领域覆盖单一**: 查询集中于 RAG 技术本身，结论可能不适用于医疗、法律、金融等专业领域。
3. **人工评分主观性**: Relevance 采用 1-5 量表，不同评分者之间可能存在系统性偏差（如某人倾向于给 3 分）。建议未来引入多评分者平均或使用 LLM-as-Judge 自动化评估。
4. **硬件环境差异**: 延迟数据在服务器共享环境下测得，可能受 GPU 显存争用影响。Flan-T5 的 0.23s 极低延迟也可能受到批处理优化干扰。
5. **检索质量未独立评估**: 本研究未单独测量 retrieval precision/recall/NDCG，无法区分检索错误和生成错误。
6. **缺乏消融实验**: 未对比"无检索直接生成"的基线，无法量化 RAG 架构本身带来的提升。

---

## 6. Conclusion and Future Work

### 6.1 Summary

本研究系统评估了分块策略（256 vs 512 tokens，10% overlap）和 LLM 后端（Mistral-7B-Instruct vs Flan-T5-base）对 RAG 文档问答系统性能的影响。基于 20 个测试问题和 4 种配置的实验结果表明：

1. **LLM 选择是决定性因素**: Mistral-7B 在 Relevance（3.85 vs 2.10）和 Task Completion（67.5% vs 0%）上全面超越 Flan-T5。Flan-T5-base 因缺乏指令微调，几乎无法完成任何问答任务。
2. **Chunk size 的影响有限但有意义**: 512 tokens 在质量和延迟上均优于 256 tokens（Relevance 3.00 vs 2.95, Latency 3.47s vs 6.01s），建议作为默认分块策略。
3. **质量-速度权衡显著**: Mistral-7B 的高质量以 16 倍延迟为代价。在资源受限场景下，Flan-T5/512 可作为快速但低质量的备选。
4. **最佳配置**: Mistral-7B/512 在所有质量指标上最优（Relevance 3.90, Completion 70%），且生成延迟（6.70s）低于 256 配置（11.10s）。

### 6.2 Future Directions

- **Query Rewriting**: 实现查询扩展和重写技术，提升检索召回率
- **Reranking**: 引入二阶段重排序器（Cross-Encoder），提升检索精度
- **Hybrid Retrieval**: 结合稠密（Dense）和稀疏（Sparse/BM25）检索方法
- **LLM-as-Judge**: 使用 GPT-4 等强模型进行自动评估，减少人工评分成本
- **Larger Evaluation Set**: 扩展至 100+ 查询，引入统计显著性检验
- **User Interface**: 开发轻量级 Web UI，降低非技术用户使用门槛
- **Scalability**: 优化大规模文档语料库和并发查询处理能力

---

## Appendix

### A. Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Chunker | chunk_size | 256 / 512 tokens |
| Chunker | overlap_ratio | 0.1 (10%) |
| Embedding | model | sentence-transformers/all-MiniLM-L6-v2 |
| Embedding | dimension | 384 |
| Retriever | top_k | 2 |
| LLM (Mistral) | max_tokens | 256 |
| LLM (Mistral) | temperature | 0.1 |
| LLM (Mistral) | quantization | 4-bit (optional) |
| LLM (Flan-T5) | max_tokens | 256 |
| LLM (Flan-T5) | temperature | 0.1 |

### B. Sample Responses

以下为同一问题在不同配置下的代表性回答示例：

**Query:** "What are the advantages of retrieval-augmented generation?"

| Configuration | Response Sample |
|--------------|-----------------|
| Flan-T5/256 | "RAG benefits." (极短，无实际信息) |
| Flan-T5/512 | "I don't know." (幻觉/放弃) |
| Mistral-7B/256 | "RAG offers several key advantages over traditional LLM approaches: it reduces hallucination by grounding responses in retrieved documents [C1], enables up-to-date knowledge without retraining [C2], and provides source attribution for verification [C1]. Compared to fine-tuning, RAG is more cost-effective and flexible for dynamic knowledge bases [C2]." (完整、有引用) |
| Mistral-7B/512 | 类似 256，但引用了更多文档片段（C1, C2, C3），回答更全面 |

### C. Detailed Evaluation Results

完整逐题评估结果见 `results/runs/analysis/` 目录下的 CSV 文件：
- `eval_flant5_256/eval_flant5_chunk256_scored.csv`
- `eval_flant5_512/eval_flant5_chunk512_scored.csv`
- `eval_mistral7b_256/eval_mistral7b_chunk256_scored.csv`
- `eval_mistral7b_512/eval_mistral7b_chunk512_scored.csv`

数值汇总见 `comparison_summary.csv`，可视化图表见 `analysis_report.txt`。

### D. Visualization Outputs

| File | Description |
|------|-------------|
| `radar_comparison.png` | 综合能力雷达图：五维度（Relevance, Completion, Speed, Conciseness, Retrieval）对比 |
| `relevance_comparison.png` | Relevance 评分柱状图 + 箱线图 |
| `completion_comparison.png` | Task Completion Rate 柱状图 |
| `latency_comparison.png` | 延迟分解图（Retrieval vs Generation vs Total） |
| `response_length_comparison.png` | 回答长度对比 |
| `relevance_heatmap.png` | 20 个问题 × 4 种配置的 Relevance 热力图 |
| `comparison_summary.csv` | 数值汇总表 |
| `analysis_report.txt` | 文本分析报告 |

---

*Report generated as part of the RAG-LlamaIndex project.*
