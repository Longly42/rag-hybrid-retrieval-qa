"""检索评测：Recall@K / MRR / 延迟。

为什么单独一个脚本，不塞进 search.py：
    search.py 是「人看结果」，eval.py 是「机器算数字」。混在一起，
    每次调试都要在交互模式和批量评估之间来回改参数，两件事互相打断。
    分开之后，调参的循环是：改 → 跑 eval.py → 看数字，中间不需要人眼介入。

为什么复用 scripts/search.py 的 get_index / load_corpus：
    索引缓存的判据只该有一处（大小 + mtime + 篇数）。两处各写一遍，
    迟早出现「search.py 认为缓存有效、eval.py 认为无效」的情况，
    然后你拿着两份不同的索引比指标 —— 这类灵异事件排查成本极高。
    代价是 eval.py 依赖 search.py，所以那边改名要同步。

用法（项目根目录，用 venv 的解释器）：
    python scripts/eval.py                 # 默认 k=5，读 data/eval/qa.jsonl
    python scripts/eval.py -k 10            # 改截断位置
    python scripts/eval.py --k1 1.2         # 试另一组 BM25 参数
"""

from __future__ import annotations

import argparse         #导入argparse库用于命令行参数解析
import json             #导入json库用于json文件读写
import statistics       #导入statistics库用于计算统计指标
import sys              #导入sys库用于获取命令行参数
import time         
from collections import defaultdict         #导入defaultdict库用于创建默认字典
from pathlib import Path                #导入Path库用于路径操作

ROOT = Path(__file__).resolve().parents[1]   #获取当前文件所在目录的父目录的父目录，即项目根目录
sys.path.insert(0, str(ROOT))                #将项目根目录添加到sys.path中，使得可以导入项目中的模块

from scripts.search import get_index, load_corpus  # noqa: E402，导入search.py中的get_index和load_corpus函数
from src.retrieval.bm25 import tokenize  # noqa: E402，导入tokenize函数
from src.retrieval.vector import VectorIndex  # noqa: E402，向量检索（模型懒加载，import 本身不重）
from src.retrieval.fuse import RRFFusion  # noqa: E402，RRF 融合

CORPUS = ROOT / "data" / "corpus.jsonl"   #定义语料库文件路径
EVAL = ROOT / "data" / "eval" / "qa.jsonl"   #定义评测集文件路径

def load_eval(path: Path) ->list[dict]:
    """读评测集，并且把所有能提前发现的错误都拦在开跑之前。

    为什么在这里较真：
        评测要跑 20 条、每条查一次，中途因为某行少个字段崩掉，
        前面跑的结果全丢，还得重来。更糟的是「静默错误」——
        relevant 写了个语料里不存在的 chunk_id，Recall 永远算不到，
        看起来就是「这条没命中」，你会去调检索参数，而问题在标注里。
    """
    if not path.exists():
       sys.exit(f"评测集不存在：{path}")

    items: list[dict] = []   #定义一个空列表用于存储评测集数据
    with path.open(encoding="utf-8") as f:   #打开评测集文件
        for ln,line in enumerate(f ,1):    #遍历评测集文件中的每一行
            line =line.strip()   #去掉行首行尾的空白字符
            if not line:   #如果行内容为空，则跳过
                continue
            try:
                r = json.loads(line)   #将行内容解析为json格式
            except json.JSONDecodeError  as e:   #如果解析失败，则抛出异常
                sys.exit(f"第 {ln} 行不是合法 JSON：{e}")
            for field in ("qid", "query", "relevant"):   #遍历评测集数据中的每个字段
                if field not in r:   #如果字段不存在，则抛出异常
                    sys.exit(f"第 {ln} 行缺少字段 {field}")
            if not r["relevant"]:
                # 分母为 0 的 Recall 没有意义，留到运行时除零更难查
                sys.exit(f"{r['qid']} 的 relevant 是空的，这条没法算 Recall")
            items.append(r)   #将评测集数据添加到列表中

    qids = [r["qid"] for r in items]   #获取评测集数据中的qid字段
    if len(qids) != len(set(qids)):   #如果qid字段有重复，则抛出异常
        sys.exit("评测集中有重复的 qid")

    # 悬空引用检查：标注里的 chunk_id 必须真的存在于语料
    known = {r["chunk_id"] for r in load_corpus(CORPUS)}   #获取语料库中的chunk_id字段
    for r in items:   #遍历评测集数据中的每一条数据
        dangling =[c for c in r["relevant"] if c not in known]   #获取评测集数据中不存在的chunk_id字段
        if dangling:
            sys.exit(f"{r['qid']} 的 relevant 引用了不存在的 chunk_id：{dangling}")
    return items   #返回评测集数据列表

def build_retriever(args, rows: list[dict], corpus_path: Path):
    """按 --mode 返回一个「有 search(query, k) 的东西」。

    三种模式返回的是完全不同的类型：BM25Index / VectorIndex / RRFFusion，
    但 evaluate() 一行都不用改 —— 这就是把签名对齐的回报。
    """
    if args.mode == "bm25":
        idx, ms, cached = get_index(rows, corpus_path, args.k1, args.b, rebuild=args.rebuild)
        print(f"BM25 语料 {len(rows)} 篇 / 词表 {len(idx.postings)} / k1={idx.k1} b={idx.b}")
        print(f"  索引{'来自缓存' if cached else '已重建'}，{ms:.0f} ms")
        return idx

    if args.mode == "vector":
        vi = VectorIndex.load()
        print(f"向量表 {vi.emb.shape[0]} 行 × {vi.emb.shape[1]} 维")
        return vi

    idx, ms, cached = get_index(rows, corpus_path, args.k1, args.b, rebuild=args.rebuild)
    vi = VectorIndex.load()
    print(f"BM25 词表 {len(idx.postings)} / 向量表 {vi.emb.shape[0]} 行")
    print(f"融合 RRF k={args.rrf_k}，每路取 {args.depth} 条")
    return RRFFusion([idx, vi], k=args.rrf_k, depth=args.depth)


def evaluate(idx, items: list[dict], k: int) -> list[dict]:
    """跑完所有 query，返回逐条明细。

    Recall 的分母是 len(relevant)，不是 1：
        一题多答案的条目（q006 / q009 / q011 / q014）就是要考「能不能全捞到」，
        只中一条就给满分，会把召回能力算虚高。
    MRR 用「第一个命中的名次」：
        一题多答案时，只要有一条排在前面就算这个题答对了位置。
    """
    per: list[dict] = []   #定义一个空列表用于存储每条评测结果
    for it in items:
        t0 =time.perf_counter()   #获取当前时间
        hits = [cid for cid, _ in idx.search(it["query"], k=k)] #使用idx.search函数对每个查询进行检索，并获取前k个命中的chunk_id
        ms =(time.perf_counter() - t0) * 1000   #计算检索时间

        gold =set(it["relevant"])   #获取评测集数据中的relevant字段
        recall = len(gold & set(hits)) / len(gold)   #计算召回率
        rank = next((i for i, cid in enumerate(hits, 1) if cid in gold), 0)  #计算第一个命中的名次

        per.append({
            "qid": it["qid"],
            "kind": it.get("kind", "?"),
            "recall": recall,
            "rank": rank,
            "mrr": 1.0 / rank if rank else 0.0,
            "ms": ms,
        })
    return per

def p95(values: list[float]) -> float:
    """
    p95 是一种统计指标，表示在所有数据中，有 95% 的数据小于等于这个值。
    该函数用于计算给定列表中数据的 p95 值。
    """
    s =sorted(values)   #对列表进行排序
    return s[min(len(s) - 1, round(0.95 * (len(s) - 1)))] #返回排序后列表中第95%位置的值

def report(per: list[dict], k: int) -> None: #定义一个函数用于生成报告
    print(f"\n{'qid':<6}{'kind':<12}{'命中':<6}{'首中名次':<9}{'Recall':<9}{'耗时ms'}")
    print("-" * 52)
    for r in per:       #遍历每条评测结果
        print(f"{r['qid']:<6}{r['kind']:<12}"
              f"{'是' if r['rank'] else '否':<6}"
              f"{r['rank'] or '-':<9}{r['recall']:<9.2f}{r['ms']:.1f}")

    # 分组比总数有用：它告诉你「哪类查询在拖后腿」，直接指向下一步该做什么。

    print("\n按 kind 分组：")
    print(f"  {'kind':<12}{'条数':<6}{'命中':<6}{'Recall@' + str(k):<11}{'MRR'}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in per:
        groups[r["kind"]].append(r)
    for kind, rs in sorted(groups.items()):
        hit = sum(1 for r in rs if r["rank"])
        print(f"  {kind:<12}{len(rs):<6}{hit:<6}"
              f"{statistics.fmean(r['recall'] for r in rs):<11.3f}"
              f"{statistics.fmean(r['mrr'] for r in rs):.3f}")

    recall = statistics.fmean(r["recall"] for r in per)
    mrr = statistics.fmean(r["mrr"] for r in per)
    hit = sum(1 for r in per if r["rank"])
    lat = [r["ms"] for r in per]

    print("\n" + "=" * 52)
    print(f"条数 {len(per)}   Top-{k} 有命中 {hit} 条")
    print(f"Recall@{k} = {recall:.4f}")
    print(f"MRR       = {mrr:.4f}")
    # 20 条的 p95 只是粗估（实际约等于第 19 名），样本量上去才有意义
    print(f"延迟 中位 {statistics.median(lat):.1f} ms   p95 {p95(lat):.1f} ms")
    print("=" * 52)


def main()-> None:
    """
    ap指针解析器，用于解析命令行参数
    """
    ap = argparse.ArgumentParser(description="评测：Recall@K / MRR / 延迟")
    ap.add_argument("-k", type=int, default=5, help="截断位置，默认 5")
    ap.add_argument("--eval", default=str(EVAL), help="评测集路径")
    ap.add_argument("--corpus", default=str(CORPUS), help="语料路径")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--mode", choices=("bm25", "vector", "rrf"), default="bm25",
                    help="检索模式：纯 BM25 / 纯向量 / 两路 RRF 融合")
    ap.add_argument("--depth", type=int, default=50, help="RRF 每路取多少条参与融合")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF 的 k 常数，一般不动")
    ap.add_argument("--rebuild", action="store_true", help="忽略缓存重建索引")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)  #获取语料路径
    items = load_eval(Path(args.eval)) #获取评测集数据
    rows = load_corpus(corpus_path)   #获取语料库数据

    retriever = build_retriever(args, rows, corpus_path)
    print(f"模式 {args.mode}")

    # 不预热的话 jieba 首次加载的 0.4s、模型首次 encode 的 5s，
    # 都会记在第一条 query 上，那条耗时直接污染 p95
    tokenize("预热")
    retriever.search("预热", 1)

    report(evaluate(retriever, items, args.k), args.k)


if __name__ == "__main__":
    main()


    