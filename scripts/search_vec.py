"""手工看向量检索结果。

    python scripts/search_vec.py              # 跑三条 symptom 查询，再进交互
    python scripts/search_vec.py "查询" -k 10
    python scripts/search_vec.py "查询" --instruction   # 查侧加 bge 指令，A/B 用

为什么默认先跑症状类：
    BM25 在这一类上 6 条只中 1 条（Recall 0.167）。查询是「现象描述」
    （改 DEBUG 之后报 DisallowedHost），文档写的是「参数语义」
    （ALLOWED_HOSTS 不匹配则 get_host 抛 SuspiciousOperation）。
    字面几乎不重合，正是向量该接管的地方 —— 看这三条比看总分直观。
    打 ★ 的是评测集里的标注答案，直接看它有没有被捞上来。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.search import load_corpus, snippet          # noqa: E402
from src.retrieval.vector import (                       # noqa: E402
    BGE_QUERY_INSTRUCTION, MODEL_NAME, VectorIndex,
)

CORPUS = ROOT / "data" / "corpus.jsonl"
EVAL = ROOT / "data" / "eval" / "qa.jsonl"

# 三条 symptom，字符串必须和 qa.jsonl 里的 query 一字不差，否则 ★ 标不出来
SYMPTOM_QUERIES = (
    "Django 接了公司单点登录，加了 RemoteUserMiddleware，但管理员登录后台进不去了",
    "用 csv 模块写出来的文件，每行数据中间多了一个空行，Windows 上特别明显",
    "把 DEBUG 改成 False 之后，本机访问直接报 DisallowedHost，之前明明能跑",
)


def load_gold(path: Path) -> dict[str, set[str]]:
    """query -> 标注的 chunk_id 集合，用来在结果里打 ★。"""
    out: dict[str, set[str]] = {}
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            r = json.loads(ln)
            out[r["query"]] = set(r["relevant"])
    return out


def show(query: str, vindex: VectorIndex, by_id: dict[str, dict],
         gold: dict[str, set[str]], k: int, instruction: str | None = None) -> None:
    t0 = time.perf_counter()
    vec = vindex.encode([query], instruction=instruction)[0]
    t1 = time.perf_counter()
    hits = vindex.search_by_vector(vec, k)
    t2 = time.perf_counter()

    print(f"\n查询：{query}")
    print(f"  编码 {(t1 - t0) * 1000:.1f} ms / 打分 {(t2 - t1) * 1000:.1f} ms")

    want = gold.get(query, set())
    if want:
        print(f"  标注答案 {len(want)} 条")

    if not hits:
        print("  无命中")
        return

    for rank, (cid, score) in enumerate(hits, 1):
        r = by_id.get(cid, {})
        mark = "★" if cid in want else " "
        print(f" {mark}[{rank}] {score:.4f}  {cid}")
        print(f"        {r.get('title_path', '')}")
        print(f"        {snippet(r.get('text', ''))}")

    if want:
        got = sum(1 for cid, _ in hits if cid in want)
        print(f"  ★ 命中 {got}/{len(want)}")


def repl(vindex: VectorIndex, by_id: dict[str, dict], gold: dict[str, set[str]],
         k: int, instruction: str | None) -> None:
    print("\n" + "=" * 58)
    print("输入查询回车检索    :k 10 改返回条数    :q 退出")
    print("=" * 58)
    while True:
        try:
            raw = input("\n查询> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not raw:
            continue
        if raw in (":q", "quit", "exit", "q"):
            return
        if raw.startswith(":k"):
            parts = raw.split()
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
                k = int(parts[1])
                print(f"返回条数改为 {k}")
            else:
                print("用法：:k 10")
            continue
        show(raw, vindex, by_id, gold, k, instruction)


def main() -> None:
    ap = argparse.ArgumentParser(description="向量检索（暴力精确，内积即余弦）")
    ap.add_argument("query", nargs="?", default=None, help="留空则跑示例并进交互")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--eval", default=str(EVAL))
    ap.add_argument("--instruction", action="store_true",
                    help="查询加 bge 官方指令前缀（A/B 用，默认关）")
    args = ap.parse_args()

    rows = load_corpus(CORPUS)
    by_id = {r["chunk_id"]: r for r in rows}

    t0 = time.perf_counter()
    vindex = VectorIndex.load(model_name=args.model)
    print(f"向量表 {vindex.emb.shape[0]} 行 × {vindex.emb.shape[1]} 维，"
          f"加载 {(time.perf_counter() - t0) * 1000:.0f} ms")

    gold = load_gold(Path(args.eval))
    instruction = BGE_QUERY_INSTRUCTION if args.instruction else None

    # 先探一次，把模型加载的 5 秒花在这里 —— 否则它会记在第一条查询的耗时上，
    # 而那个数字是要拿去看延迟的。
    vindex.encode(["预热"])

    if args.query:
        show(args.query, vindex, by_id, gold, args.k, instruction)
        return

    for q in SYMPTOM_QUERIES:
        show(q, vindex, by_id, gold, min(args.k, 5), instruction)

    repl(vindex, by_id, gold, args.k, instruction)


if __name__ == "__main__":
    main()
