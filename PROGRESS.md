# PROGRESS.md —— 进度与决策交接文档

> 这份文件是给「下一个人（或下一个 AI 会话）」看的。它记录事实、决策和踩过的坑，不负责好看。
> 对外介绍看 [README.md](README.md)，指标流水账看 [experiments.md](experiments.md)。

最后更新：2026-11-20 11:00

---

## 0. 给接手的人

### 0.1 铁律

**命令必须给 PowerShell 版本。** 环境是 Windows + VSCode + PowerShell，不是 Git Bash。曾给过 `source .venv/Scripts/activate` 导致走弯路。

### 0.2 三十秒核实进度（不要相信第 2 节，跑一遍）

```powershell
git log --oneline
git status --short
(Get-Content data\urls.txt | Measure-Object -Line).Lines      # 链接数
(Get-ChildItem data\raw -Filter *.html).Count                  # 已抓篇数
test -f data\corpus.jsonl; (Get-Content data\corpus.jsonl | Measure-Object -Line).Lines
Get-ChildItem src,scripts -Recurse -Filter *.py | Select Name,Length
Get-Content experiments.md
```

### 0.3 用户水平

- **已确认**：能独立写 Python 脚本和爬虫（requests + 解析 + 落盘），报错会自己查。不讲「怎么装包」这类基础。
- **未确认**：检索原理的数学直觉、FastAPI / 数据库 / 并发、C++ 实际动手经验。这三块按零基础讲，但要短、要落在代码上。
- **沟通偏好**：直接指出错误，不要客套，不要先夸再批。错在哪、为什么错、正确路径是什么，三句话说清。
- **典型倾向（已出现两次，需持续纠正）**：偏向「先搭好门面」——写 README 比写代码积极、把未完成的东西描述成已完成、用忽略规则掩盖问题而不去修它。每次发现都要立刻按住。

---

## 1. 项目目标

### 1.1 目标规格（来源：本人简历，属「目标」不属「已达成」）

**项目一 —— 混合检索与重排序的 RAG 问答系统**

- 多格式解析（PDF / Word / Markdown）→ 按标题层级切分并控制重叠 → 向量化入库（Chroma）
- 目标规模：约 3k 篇文档 / 3 万 chunk
- 生成侧 SSE 流式输出，回传命中片段支撑引用溯源
- 检索优化：纯向量召回不足 → BM25 + 向量混合（RRF）→ Top-50 后接 Cross-Encoder（bge-reranker-base）重排
- 量化评估：120 条人工标注评测集，Recall@K / MRR 自动化回归，三组配置对比
- 延迟权衡：p95 210ms → 380ms → 压回 260ms
- 工程落地：MySQL 会话/消息表支撑多轮回溯，FastAPI 异步串联检索与生成

**项目二 —— RAG 检索层下沉：C++ BM25 倒排索引内核**

- 动机：Python 检索层在 3 万篇规模 p95 达 400ms，线性劣化
- 实现：词项 → posting list；连续数组存储；BM25 归一化；top-k 用大小为 k 的最小堆剪枝；索引只读、查询侧无锁多线程
- 验证：与 Python 版 top-10 一致率 100%（浮点误差 < 1e-6，并列按 doc_id 稳定排序）；12 万篇规模 p95 148ms → 19ms（8 线程只读）
- 已知局限：索引需全量重建，未支持增量更新与磁盘持久化

### 1.2 两个项目的关系（最容易搞错的地方）

**不是两个独立项目，是一条主线加一个支线。**

必须先有 Python 版 BM25 和可复现的延迟基线，C++ 内核才有「优化目标」和「正确性对照物」。顺序反了，C++ 写完也无法证明它是对的、也无法证明它变快了。

---

## 2. 当前进度（2026-11-20 11:00）

### 2.1 已完成

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 目录骨架 | 完成 | `data/` `src/ingest/` `src/retrieval/` `scripts/`，各包有 `__init__.py` |
| 依赖清单 | 完成 | requests / beautifulsoup4 / lxml / jieba（**刻意没有** numpy、pandas、任何向量库） |
| 虚拟环境 | 完成 | 目录名 `venv`（不是 `.venv`），已在 `.gitignore` 中 |
| Git | 完成 | 3 次提交，已推送 `origin/main` → github.com/Longly42/rag-hybrid-retrieval-qa |
| 忽略规则 | 完成 | `venv/`、`data/`、`new.py` 均已覆盖（`new.py` 后来已物理删除） |
| `crawl.py` | 骨架完成 | `--discover` / `--limit N` / 断点续爬已写好，**待填 `CATALOG_URL` 与 `LINK_SELECTOR`** |
| 交接文档 | 完成 | 本文件 |

### 2.2 已清理的历史问题（不要再犯）

1. **把目录说明写成文件内容。** `data/raw` 曾是一个 37 字节的文件，里面是「爬下来的原始 HTML」这行注释；`data/corpus.jsonl` 曾是一个 43 字节的说明文字。根因是误解「git 不跟踪空目录」，于是用写文件的方式占位。**正确做法：目录就是目录，要占位放 `.gitkeep`。**
2. **用忽略规则掩盖问题。** `new.py` 被写进 `.gitignore` 而不是删除。忽略只是让它不显示，问题还在。
3. **`.gitignore` 与实际目录名不一致。** 写的 `.venv/`，实际叫 `venv/`。
4. **提前建空文件造成「已经开始了」的错觉。** `bm25.py` / `build_index.py` / `eval.py` / `search.py` 曾是 0 字节空壳，已删。约定：写到哪建到哪。
5. **把未完成的成果写成已完成。** README 曾被改成一版「规模实测 3k 篇 / Recall@5 65%→88% / 引用准确率 92%」的稿子，而当时仓库里 0 篇文档、0 行检索代码。已改为「目标 / TBD」，并拆出本文件承接交接。

### 2.3 下一步（唯一优先级）

**填 `crawl.py` 的两个 TODO，跑 `--discover`，用链接数决定站点去留，然后抓 5 篇验收。**

1. 选定目标文档站（判据见 ADR-7），填 `CATALOG_URL`
2. F12 看目录页结构，填 `LINK_SELECTOR`
3. `python -m src.ingest.crawl --discover` → 看输出条数
   - **< 800 条**：选择器抽错了，或站点撑不起规模，换站
   - **≥ 800 条**：继续
4. `python -m src.ingest.crawl --limit 5` → 抓 5 篇
5. 五项人工验收（缺一不可）：
   - 每个文件 20KB–300KB（1–3KB 说明抓到反爬页 / JS 空壳，换站，不要上 Selenium 硬啃）
   - 正文完整，没混进侧边栏 / 推荐阅读
   - 标题是 `h1/h2/h3` 语义标签，`section_path` 能提取
   - 代码块保住了
   - 中文无乱码
6. 五项全过才扩量到 1000 篇

尚未开始：任何数据抓取、任何检索代码。

---

## 3. 目录结构

```
rag-hybrid-retrieval-qa/
├── data/                 # 整个目录不入库
│   ├── raw/              # 原始 HTML，一文件一篇，文件名 = doc_id.html
│   ├── urls.txt          # 目录页探测结果，doc_id 顺序的唯一来源
│   └── corpus.jsonl      # 清洗后语料，一行一文档
├── src/
│   ├── ingest/
│   │   ├── crawl.py      # 只负责把 HTML 存下来
│   │   └── parse.py      # 只负责把 HTML 洗成 jsonl（待写）
│   └── retrieval/
│       └── bm25.py       # 检索实现（待写）
├── scripts/              # 一次性入口脚本，可随时删了重写
├── experiments.md        # 指标流水账，比代码更值钱
├── README.md             # 对外介绍
└── PROGRESS.md           # 本文件
```

### 3.1 语料格式（一行一 JSON）

```json
{"doc_id":"0001","url":"...","title":"...","section_path":["第3章 存储引擎","3.2 LSM-Tree"],"content":"正文","code":["..."],"char_count":6421}
```

- `doc_id`：定长数字字符串，**按 URL 字典序排序后编号**，保证重跑不变
- `section_path`：标题栈维护出的完整层级，决定切分质量的天花板
- `code`：单独存，便于以后做「代码块不参与 BM25 打分」的实验
- `char_count`：存下来，省得每次重算

---

## 4. 技术决策记录（ADR）

已定案，新会话不要推翻，也不要重新论证。

**ADR-1 先手写 BM25，再碰任何框架。** 不用 LangChain 起手，要求不看资料手写 60 行以内的 BM25。理由：RRF 为什么用排名倒数、重排为什么能救回 BM25 漏掉的语义匹配，全靠这一层的体感。先封装后原理，会变成只会调参的人。

**ADR-2 先 BM25，后向量。** 不先上 embedding。不知道字面匹配的失效边界，就不知道向量补的是什么。

**ADR-3 抓取与解析拆成两段跑。** `crawl.py` 只管存 HTML，`parse.py` 只管洗。清洗规则一定会改很多遍，分离后改规则不必重爬。

**ADR-4 doc_id 按 URL 字典序编号。** 不用抓取顺序，不用 uuid。要让 C++ 版与 Python 版做 top-10 一致性比对、并列时按 doc_id 稳定排序，编号必须稳定。

**ADR-5 RRF 而非分数加权。** 两路分数不同量纲（BM25 无上界，余弦在 [-1,1]），不可直接相加。RRF 只用排名，天然跨量纲。

**ADR-6 重排在 Top-50 处截断。** Cross-Encoder 逐对推理，O(N) 次模型前向。重排 10000 条与 50 条差两个数量级，这是 p95 从 380ms 压回 260ms 的主要来源。

**ADR-7 语料必须可扩到 3000 篇以上。** C++ 项目的动机是「语料增长后 Python 线性劣化」，站点太小第二个项目就没有立足点。若中文站不够，C++ 性能测试阶段可换用 arXiv 批量数据（那阶段要的是性能不是中文分词）。

**ADR-8 现在不装 numpy / pandas / 任何向量库。** 还没到那一层，装了只会诱人提前跳到阶段 ③。

**ADR-9 数据全部不入库，靠脚本重建。** 1000 篇中文语料约 50MB，3000 篇更大，12 万篇时 git 完全不适合。所以 `data/` 整体忽略，README 里给出重建步骤。等链路稳定、语料定型后，可考虑用 `!data/sample.jsonl` 白名单放 50 篇样本进仓库，让 clone 下来就能跑——现在还不必要。

---

## 5. 学习路线与验收标准

| 阶段 | 内容 | 验收（没过不进下一阶段） |
| --- | --- | --- |
| ① 工程底座 | Python 工程化、Git、依赖管理 | 换台机器能跑起来（**进行中，接近完成**） |
| ② 检索原理 | 分词、TF-IDF、BM25（k1/b）、向量与余弦 | **手写 BM25 不看资料**，1000 篇上排序合理 |
| ③ 最小 RAG | 解析 → 切分 → 向量库 → 检索 → SSE 流式 | 50 篇里问一个只有第 37 篇能答的问题，答对且给出处 |
| ④ 混合检索与重排 | BM25+向量 RRF、Cross-Encoder、评测集 | 三组配置的 Recall@5 / MRR 对比表 |
| ⑤ 服务化 | FastAPI 异步、MySQL 会话表、前端、引用对齐 | 连问三轮相关问题，第三轮仍能正确引用第一轮上下文 |
| ⑥ 评测与调优 | 120 条标注集、p95 压测、截断与批处理 | 延迟数字可复现，每次改动都有记录 |

C++ 阶段（必须在 ③④ 完成、基线可复现之后）：

| 步骤 | 内容 | 验收 |
| --- | --- | --- |
| C1 基本功 | RAII、STL、移动语义、多线程 | 能说清 `vector<vector<T>>` 与连续数组的性能差异 |
| C2 倒排索引 | posting list、连续数组 + 偏移表 | 索引构建正确，内存布局可解释 |
| C3 top-k 与并发 | **大小为 k 的最小堆**、只读索引无锁多线程 | 8 线程并发加速比接近线性 |
| C4 绑定与验证 | pybind11、**必须释放 GIL**、结果比对 | 与 Python 版 top-10 一致率 100%（`fabs(a-b) < 1e-6`，并列按 doc_id 稳定排序） |

C3 的两个必答点（面试高频，也是自己最容易写错的）：
- top-k 用**最小堆**（维护当前最大的 k 个），不是最大堆，不是全排序
- pybind11 绑定长耗时函数必须 `py::gil_scoped_release`，否则 8 线程退化成 1 线程，「19ms」就是假的

---

## 6. 开发约定

### 6.1 环境（Windows / PowerShell）

```powershell
# 激活（若报「禁止运行脚本」）
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1

# 不想碰策略，就一直用全路径，更安全
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

激活成功的标志：提示符前有 `(venv)`，且 `python -c "import sys; print(sys.prefix)"` 指向项目下的 `venv`。

### 6.2 运行入口脚本

一律用 `python -m` 从项目根目录跑，不要 `python scripts/xxx.py`：

```powershell
python -m src.ingest.crawl --discover
```

`-m` 会把项目根加入 `sys.path`，`from src.xxx import yyy` 才不会报 `ModuleNotFoundError`。这比在每个脚本顶部插 `sys.path` 干净。

### 6.3 Git

- 每完成一个「能跑的东西」就 commit，不要攒大招
- message 说清做了什么：`feat: 抓取 XX 文档 1200 篇并统一为 jsonl`
- `data/` 与 `venv/` 永不入库；不要用 `.gitignore` 掩盖该删的文件
- commit 历史就是工作证明，面试时比简历上任何一句话都硬

### 6.4 抓取礼貌

限速 `sleep(0.6~1)`，UA 带项目名与联系方式，单个失败不中断整批，遇 403/429 立刻停。内容仅用于本地实验，不公开分发。

---

## 7. 指标

`experiments.md` 是唯一的指标来源，写进简历的每个数字都必须能在这里追到一条记录。

当前状态：**无任何指标**。README 里的 88% / 260ms / 92% 全部是目标值，不是实测值。

---

## 8. 时间线

| 时间 | 目标 |
| --- | --- |
| 2026-11 | 语料库 + 手写 BM25 跑通 |
| 2026-12 ~ 2027-01 | 切分、向量库、混合检索、重排、评测集 |
| 2027-01 ~ 2027-02 | FastAPI 服务化、延迟调优、引用对齐 |
| 2027-02 ~ 2027-03 | C++ 内核、pybind11 绑定、一致性验证 |
| 2027-03 起 | 投暑期实习（大三下） |

简历上两个项目的日期需按实际开工时间改写，不能沿用草稿里的 2026.08 / 2026.10。

---

## 9. 已踩 / 将踩的坑

1. **给 bash 命令，用户在 PowerShell 里跑。** 已踩。命令一律 PowerShell 版。
2. **把说明文字写进数据文件 / 把目录建成文件。** 已踩。占位用 `.gitkeep`。
3. **`.gitignore` 与实际目录名不一致。** 已踩。
4. **提前建空文件造成「已经开始了」的错觉。** 已踩。
5. **用忽略规则代替删除。** 已踩。
6. **把未测数字写成实测。** 已踩（README 事件）。这是最危险的一类，因为它会在面试现场引爆。
7. **`python scripts/xxx.py` 导致 `import src` 失败。** 将踩，已在 6.2 规避。
8. **Cross-Encoder 重排不 batch。** 将踩。逐条推理会慢到无法接受。
9. **pybind11 未释放 GIL。** 将踩。表现是多线程完全无加速，且很难看出原因。
10. **C++ 与 Python 浮点结果不全等就断言实现错了。** 将踩。归一化顺序不同会有 1e-7 级差异，比对要用容差，并列必须按 doc_id 稳定排序。
