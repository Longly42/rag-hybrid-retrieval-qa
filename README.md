# RAG Hybrid Retrieval QA

一个面向专业文档的高精度检索增强生成（RAG）系统。通过 **BM25 + 向量双路混合检索**、**RRF 融合** 与 **Cross-Encoder 重排**，解决大模型在专业术语、代码函数及长尾知识上的幻觉问题。

## ✨ 核心特性

- **多源异构数据管道**：自研爬虫与解析器，支持 HTML、Markdown、reStructuredText 等多格式文档的结构化清洗，基于 `<section>` 递归与标题层级精准切分。
- **混合检索（Hybrid Search）**：针对纯向量检索在专有名词 / 精确关键词上召回不足的痛点，引入 BM25 与向量双路召回。
- **RRF 分数融合**：使用 Reciprocal Rank Fusion 算法解决异构检索得分量纲不一致问题，兼顾语义泛化与字面匹配。
- **Cross-Encoder 精排**：在 Top-50 处截断并接入 `bge-reranker-base` 重排模型，逐对推理打分，大幅提升 Top-K 命中率。
- **高性能底层支持（规划中）**：检索层计划下沉至 C++17 内核（连续数组存储、最小堆 top-k 剪枝、无锁并发），解决大规模语料下的延迟瓶颈。

## 📊 当前进度与指标

| 模块 | 当前状态 | 核心指标 / 备注 |
| :--- | :--- | :--- |
| **数据清洗** | ✅ 已完成 | 已处理 **16,378 个 chunk**（多源技术文档） |
| **BM25 检索** | ✅ 已完成 | 单次查询延迟 **0.4 ~ 0.8 ms**（Python 实现） |
| **向量检索** | 🚧 进行中 | 对接 Chroma DB 与 BGE Embedding 模型 |
| **融合与重排** | ⏳ 待开发 | 计划引入 RRF 与 `bge-reranker-base` |
| **生成与溯源** | ⏳ 待开发 | FastAPI + SSE 流式输出 + 引用高亮 |

*注：120 条人工评测集正在构建中，待全链路跑通后将补充 Recall@K / MRR 等完整性能对比数据。*

## 🏗️ 系统架构与链路

```text
[ 多源数据 ]  HTML / Markdown / reStructuredText
     │
     ▼
[ 语料清洗与切分 ]  BeautifulSoup · 正则去噪 · <section> 递归切分
     │
     ├─────────────────────────┐
     │                         │
     ▼                         ▼
[ 向量索引 ]              [ 倒排索引 ]
Chroma + BGE              BM25 + jieba
     │                         │
     └──────────┬──────────────┘
                ▼
      [ RRF 融合 · Top-50 ]
                │
                ▼
 [ Cross-Encoder 重排 · Top-5 ]
      bge-reranker-base
                │
                ▼
    [ LLM 生成 + 引用溯源 ]
                │
                ▼
   [ 交互层 · FastAPI + SSE ]
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建并激活虚拟环境（Windows 示例）
python -m venv venv
./venv/Scripts/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 一键跑通 BM25 检索

体验基于字面精准匹配的倒排索引检索效果：

```bash
python scripts/search.py
```

首次运行将自动构建或加载本地 BM25 索引，随后进入交互式查询模式。

```
查询> 类装饰器 property          # 直接输入就查
:k 10                           # 改返回条数
:tok                            # 开关分词预览
:q                              # 退出
```

带参数运行则只查一条、查完退出，方便写进脚本：

```bash
python scripts/search.py "SESSION_COOKIE_DOMAIN 不生效" -k 3
```

## 📂 目录结构

```text
.
├── data/
│   ├── raw/            # 原始抓取文件，按「格式/域名/路径」分层存储
│   ├── urls/           # 抓取链接清单
│   ├── manifest.jsonl  # 抓取账本，支持断点续爬与溯源
│   └── corpus.jsonl    # 清洗切分后的 chunk 语料（可重建，不入库）
├── src/
│   ├── ingest/         # 数据管道：多源抓取、格式解析、切分清洗
│   └── retrieval/      # 检索内核：手写 BM25、倒排索引、分词
├── scripts/            # 入口脚本：索引构建、检索测试、评测等
└── experiments.md      # 实验记录与指标流水账
```

## 🧠 核心设计决策

### 为什么不直接用 LangChain？

为了彻底理解 RAG 链路的每一层，本项目从零手写了 BM25 倒排索引、RRF 融合以及切分逻辑。我们认为，只有理解底层机制，才能在实际场景中做出正确的调优与权衡。

### 为什么需要混合检索？

在专业术语（如 `operator.itemgetter`）和特定错误码场景下，向量模型的 Embedding 容易发生“语义漂移”或切词破碎。引入 BM25 是为了兜住字面精确匹配的底线。

### Chunk 切分的重叠策略

基于 Sphinx 的 `<section>` 树递归提取，而非固定的字符硬切。这保证了代码块不被截断，且每个 Chunk 都自带完整的 `title_path` 上下文（如 `内置函数 > abs()`）。

如果这个项目对你有帮助，欢迎 Star ⭐️！
