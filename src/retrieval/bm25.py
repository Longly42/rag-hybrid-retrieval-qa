"""BM25：倒排索引 + Okapi 打分，不依赖 rank_bm25。

为什么不直接装 rank_bm25：
    这个项目的定位是「每一层都能讲清楚在干什么」。BM25 就四行公式，
    装个库省下的时间，会在调参（k1 / b / 分词）时连本带利还回去

打分公式（Okapi BM25，idf 用 +1 的对数形式）：

    score(q, d) = Σ_t  idf(t) · qtf(t) · tf(t,d)·(k1+1)
                              ────────────────────────────────────────
                              tf(t,d) + k1·(1 - b + b·|d|/avgdl)

    idf(t) = ln( 1 + (N - df + 0.5) / (df + 0.5) )

为什么 idf 要加那个 1：
    经典写法是 ln((N-df+0.5)/(df+0.5))，一旦某个词出现在超过一半的文档里，
    分子小于分母，idf 变负 —— 于是「含有这个词」反而被扣分，排序开始倒挂。
    加了 1 之后 idf 恒为正，df 再大也只是趋近于 0，不会惩罚命中。
    中文语料里高频词（「使用」「函数」）很容易过 N/2，这不是理论洁癖。

为什么 k1=1.5、b=0.75：
    这是 Okapi 原论文在 TREC 上的一组经验值，也是绝大多数实现的默认。
    k1 控制「词频饱和速度」，b 控制「长度归一化的力度」。
    这两个数该在自己的评测集上调，但在有评测集之前，先用默认值，别拍脑袋。

倒排索引怎么存（这是这个文件里唯一需要动脑子的地方）：
    postings: dict[str, array('i')]，term → 文档号序列，**同一个文档号重复 tf 次**。
        例：「cookie」出现在 3 号文档 2 次、9 号文档 1 次
            → array('i', [3, 3, 9])
        这样 tf 不用单独存，数一数连续相同值的个数就是 tf。
        16k 个 chunk、约 200 万个 posting，用 array('i') 是 8MB；
        换成 list[tuple[int,int]] 要 200MB 上下 —— 差一个数量级。
        因为构建时按文档号递增顺序扫，array 天然有序，查询时顺序扫一遍就是并集。

    df（文档频率）单独用 dict[str,int] 记：array 的长度是 posting 总数（Σtf），
    不是 df。混用的话 idf 会算成 «词频越高 idf 越小»，整个打分就废了。

分词（中文是这个项目里比 BM25 本身更影响效果的一层）：
    - 中日韩字符走 jieba 的 search 模式（切得细，长词同时产出子词，召回高）
    - 英文/数字串按 _ . - 拆开：「SESSION_COOKIE_DOMAIN」→ session / cookie / domain
      代码文档里同义词往往就差个下划线，不拆开就召不回来
    - 纯数字串（如 5.2）整体保留：拆成 5 和 2 是纯噪声
    - 单字中文、单字母英文一律丢：它们几乎不携带区分度，只贡献噪声分数

用法：
    from src.retrieval.bm25 import BM25Index
    idx = BM25Index()
    idx.build(((c["chunk_id"], c["text"]) for c in chunks))
    for chunk_id, score in idx.search("SESSION_COOKIE_DOMAIN 不生效", k=5):
        ...
"""

from __future__ import annotations

import heapq
import logging
import math
import pickle
import re
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import jieba

# jieba 首次加载词典会在 stderr 打日志，CLI 里看着像报错，压到 WARNING
jieba.setLogLevel(logging.WARNING)

# ---------- 分词配置 ----------

# 中日韩统一表意文字：基本区、扩展 A、兼容区
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# 一次扫出「中文串」或「英文/数字串」。英文串允许中间夹 _ . -
_TOKEN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
                       r"|[A-Za-z0-9]+(?:[._\-][A-Za-z0-9]+)*")
_SPLIT_RE = re.compile(r"[._\-]+")
# 纯数字（含小数/版本号），整体保留，不拆
_NUM_RE = re.compile(r"^[0-9]+(?:[._\-][0-9]+)*$")

# 停用词：只收「高频且无区分度」的虚词，不要贪多。
# 中文里多收一个实词（比如「使用」），就等于把这个词的召回整批删掉。
_STOPWORDS = frozenset("""
的 了 和 与 或 是 在 有 不 也 就 都 而 及 等 被 把 对 从 到 为 以 于 之 其 这 那 你 我 他
我们 你们 他们 它们 一个 一些 什么 怎么 如何 可以 需要 使用 进行 通过 但是 因为 所以 如果
the a an and or of to in on at for with is are was were be been being this that these those
it its as by from we you they he she not no do does did can could should would will may might
""".split())


def tokenize(text: str) -> list[str]:
    """文本 → 检索词列表。索引与查询必须用同一个函数，否则分数没法比。"""
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group()
        if _CJK_RE.match(tok[0]):          # 中文串交给 jieba
            for w in jieba.cut_for_search(tok):
                if len(w) >= 2 and w not in _STOPWORDS:
                    out.append(w)
        elif _NUM_RE.match(tok):           # 5.2 / 3.11 整体保留
            out.append(tok)
        else:                              # 英文标识符按分隔符拆开
            for part in _SPLIT_RE.split(tok.lower()):
                if len(part) >= 2 and part not in _STOPWORDS:
                    out.append(part)
    return out


class BM25Index:
    """BM25 倒排索引。build 一次，可以反复 search。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []                  # 文档号 → chunk_id
        self.postings: dict[str, array] = {}      # term → 文档号序列（重复 tf 次）
        self.df: dict[str, int] = {}              # term → 含该词的文档数
        self.lengths: list[int] = []              # 文档号 → 长度（词数）
        self.avgdl: float = 0.0
        self._idf: dict[str, float] = {}          # idf 缓存：同一 term 每次查询都要用

    # ---------- 构建 ----------

    def build(self, docs: Iterable[tuple[str, str]]) -> "BM25Index":
        """docs: (chunk_id, text) 的可迭代对象。边读边建，不要求一次读进内存。"""
        postings = self.postings
        df = self.df
        lengths = self.lengths
        ids = self.ids

        for idx, (chunk_id, text) in enumerate(docs):
            ids.append(chunk_id)
            toks = tokenize(text)
            lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                arr = postings.get(term)
                if arr is None:
                    arr = postings[term] = array("i")
                    df[term] = 1
                else:
                    df[term] += 1
                arr.extend([idx] * tf)            # 文档号递增 → 天然有序
            if (idx + 1) % 5000 == 0:
                print(f"  已索引 {idx + 1} 篇…")

        self.avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0
        self._idf.clear()
        return self

    # ---------- 打分 ----------

    def idf(self, term: str) -> float:
        """ln(1 + (N-df+0.5)/(df+0.5))，恒正，见文件头说明。"""
        v = self._idf.get(term)
        if v is None:
            n = len(self.ids)
            df = self.df.get(term, 0)
            v = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            self._idf[term] = v
        return v

    @staticmethod
    def _groups(arr: array) -> Iterator[tuple[int, int]]:
        """有序文档号序列 → (文档号, tf)。连续相同值的个数就是 tf。"""
        prev = -1
        tf = 0
        for d in arr:
            if d == prev:
                tf += 1
            else:
                if tf:
                    yield prev, tf
                prev = d
                tf = 1
        if tf:
            yield prev, tf

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score)]，按分数降序，只含有命中的文档。"""
        if not self.ids or not self.avgdl:
            return []

        qtf = Counter(tokenize(query))
        if not qtf:
            return []

        scores = [0.0] * len(self.ids)
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for term, qn in qtf.items():
            arr = self.postings.get(term)
            if not arr:
                continue
            # 查询词频也计入：长查询里重复出现的词，说明它更要紧
            w = self.idf(term) * qn
            for doc, tf in self._groups(arr):
                dl = self.lengths[doc]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[doc] += w * tf * (k1 + 1.0) / denom

        top = heapq.nlargest(k, range(len(scores)), key=scores.__getitem__)
        return [(self.ids[i], scores[i]) for i in top if scores[i] > 0.0]

    # ---------- 存取 ----------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump({
                "k1": self.k1, "b": self.b, "ids": self.ids,
                "postings": self.postings, "df": self.df,
                "lengths": self.lengths, "avgdl": self.avgdl,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as f:
            d = pickle.load(f)
        idx = cls(k1=d["k1"], b=d["b"])
        idx.ids, idx.postings, idx.df = d["ids"], d["postings"], d["df"]
        idx.lengths, idx.avgdl = d["lengths"], d["avgdl"]
        return idx

    def stats(self) -> dict:
        """规模自检用：这几个数应该和你报告的篇数、chunk 数对得上。"""
        return {
            "docs": len(self.ids),
            "terms": len(self.postings),
            "postings": sum(len(a) for a in self.postings.values()),
            "avgdl": round(self.avgdl, 1),
        }
