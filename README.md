# RAG 混合检索问答系统

> 基于 BM25 + 向量混合检索（RRF 融合）与 Cross-Encoder 重排的检索增强问答系统。

**状态：开发中。** 2026-11-20 起从零实现。当前仓库里只有工程骨架和抓取脚本，**没有检索代码，没有任何实测数据**。

进度、技术决策、踩坑记录见 [PROGRESS.md](PROGRESS.md)——那份文件是给「下一个人（或下一个 AI 会话）」看的，这份是给外面看的。

## 目标链路

一条完整的 RAG 链路，从文档解析到流式问答与引用溯源：

- **入库**：多格式解析（PDF / Word / Markdown）→ 按标题层级切分并控制重叠 → 向量化写入 Chroma
- **检索**：BM25 与向量检索并行召回 → RRF 融合 → Top-50 交给 Cross-Encoder（bge-reranker-base）重排
- **生成**：SSE 流式输出，回传命中片段支撑引用溯源
- **会话**：MySQL 存储会话 / 消息表，支撑多轮上下文与回溯

## 为什么做混合检索

纯向量检索在专业术语、专有名词、精确关键词场景下召回不足——语义泛化是优点，但在「必须字面命中」的查询上就是短板。因此引入 BM25 与向量双路召回，用 RRF 做排名融合，再交 Cross-Encoder 精排，兼顾语义泛化与字面匹配。

RRF 只用排名不用分数，是因为两路分数不同量纲（BM25 无上界，余弦相似度在 [-1, 1]），直接加权相加没有可比性。重排在 Top-50 处截断，是因为 Cross-Encoder 是逐对推理，复杂度是 O(N) 次模型前向。

## 目标指标（尚未实测，当前一律为 TBD）

建 120 条人工标注评测集，Recall@K / MRR 自动化回归，对比三组配置（纯向量 / 纯 BM25 / 混合 + 重排）。

| 指标 | 基线（纯向量） | 目标 | 现状 |
| --- | --- | --- | --- |
| Recall@5 | 待测 | ≥ 88% | **TBD** |
| p95 延迟 | 待测 | ≤ 260 ms（重排未优化时可能到 380 ms） | **TBD** |
| 引用准确率（人工抽检） | — | ≥ 92% | **TBD** |

延迟压回的手段（计划）：Top-50 截断、重排 batch 化、chunk 预取。每次策略调整都记录效果与延迟的取舍，不只是效果单边决策。

> 这一节的数字只有在 [experiments.md](experiments.md) 里有对应记录时才算数。写下没测过的数字，等于给自己埋一颗面试时会炸的雷。

## 技术栈

| 阶段 | 已用 | 计划引入 |
| --- | --- | --- |
| 抓取与解析 | requests、BeautifulSoup、lxml | — |
| 分词 | jieba | — |
| 检索 | 手写 BM25（Python） | Chroma、bge-reranker-base |
| 服务 | — | FastAPI、MySQL |
| 性能 | — | C++17、pybind11 |

刻意不上 LangChain 起手：先手写检索，理解每一层在做什么，再考虑封装。

## 目录结构（当前真实状态）

```
.
├── data/
│   ├── raw/            # 原始文件，按「格式/域名/源路径」三层存（不入库）
│   │                   #   html/docs.python.org/3/library/bisect.html
│   │                   #   html/docs.djangoproject.com/zh-hans/5.2/topics/db/models.html
│   │                   #   md/github.com/jackfrued__Python-100-Days/Day01-20/01.初识Python.md
│   ├── samples/rst/    # 3 篇 rst 原文，只用来测 formats.py 的 rst 解析（入库）
│   ├── urls/           # 每个抓取源一份链接清单，互不覆盖（入库）
│   ├── urls.txt        # 上面所有清单的并集，供 parse 校验（入库）
│   ├── manifest.jsonl  # 抓取账本：请求地址 → 落地路径，断点续爬靠它（入库）
│   └── corpus.jsonl    # 清洗后语料，一行一 chunk（不入库，可重建）
├── src/
│   ├── ingest/
│   │   ├── crawl.py    # 多源抓取：目录页 / GitHub 仓库 / Sphinx 源码
│   │   ├── parse.py    # 统一调度，打包成 chunk
│   │   ├── formats.py  # 格式探测 + rst/md/纯文本解析
│   │   └── layout.py   # raw 的目录约定，crawl 与 parse 共用一个实现
│   └── retrieval/      # bm25.py 检索实现（待写）
├── scripts/            # 一次性入口脚本（含 raw 目录迁移）
├── experiments.md      # 指标流水账
└── PROGRESS.md         # 交接文档
```

`doc_id` 由「域名 + 路径」派生，锚在 URL 上，不再依赖文件在清单里排第几——早期用位置编号，重爬一次就整批错位，这个坑已经踩过两次。

`src/` 放可复用库，`scripts/` 放入口脚本。抓下来的原始文件和切好的语料都不入库，靠 `crawl.py` + `parse.py` 重建——原因见 PROGRESS.md 的 ADR-9；清单和账本体积小、有溯源价值，进 Git。

## 相关

检索层计划下沉为 C++17 内核（posting list 连续数组存储、最小堆 top-k、只读无锁并发、pybind11 绑定）。

**独立仓库尚在规划中，未开始。** 在它真正有代码之前，这里不放链接。
