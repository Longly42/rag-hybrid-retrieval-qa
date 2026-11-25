"""检索入口：直接运行就有结果，不用记参数。

    python scripts/search.py            # 建/读索引 → 跑 3 条示例查询 → 进交互模式
    python scripts/search.py "查询"      # 只跑这一条，查完退出（脚本/批处理用）
    python scripts/search.py "查询" -k 3 --site docs.python.org

为什么不带参数也要能跑：
    检索这种步骤，用的频率远高于抓取。要求每次都想一遍 -k / --site / --k1，
    最后的结果就是「算了，直接改代码试」。所以默认值全部内置，
    交互模式里用 :k / :tok 现调，调好了再回命令行一次性固化。

索引缓存（data/index/bm25_k1-*_b-*.pkl）：
    16k chunk 建一次 4.5 秒，每查一句都重付这 4.5 秒，交互模式基本没法用。
    所以第一次运行建完就 pickle 落盘，之后直接 load（30 毫秒）。
    文件名带 k1/b，调参时每组参数一份缓存，A/B 切换不用来回重建。

    缓存是否有效不靠「文件存在」判断 —— 那是最容易出灵异事件的做法：
    重新切过语料，索引却还是旧的，分数看着就不对劲，还查不出原因。
    判据是语料指纹（大小 + mtime_ns）+ 篇数，任一项对不上就重建。

    语料被重切但大小和 mtime 恰好都没变（几乎不可能，但不是零），缓存会误判有效。
    真要干净结果：--rebuild。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25 import BM25Index, tokenize  # noqa: E402

CORPUS = ROOT / "data" / "corpus.jsonl"
INDEX_DIR = ROOT / "data" / "index"

# 三条示例覆盖三种典型查询：英文标识符 / 中文概念 / 代码术语。
# 挑这三条是为了第一眼就能看出「分词有没有切坏」——英文串要拆开、中文要成词。
SAMPLE_QUERIES = (
    "SESSION_COOKIE_DOMAIN 不生效",
    "类装饰器 property",
    "事务回滚 atomic",
)


def load_corpus(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def snippet(text: str, limit: int = 110) -> str:
    one = " ".join(text.split())
    return one if len(one) <= limit else one[:limit] + "…"


# ---------- 索引缓存 ----------

def index_paths(k1: float, b: float) -> tuple[Path, Path]:
    """按 k1/b 分槽。

    单槽缓存调参时会来回重建：k1=1.5 建一次、改成 1.2 又建一次、改回来再建一次。
    调参本来就要反复 A/B，所以索引文件按参数命名，各存一份。
    代价是 data/index/ 会堆几个文件 —— 不入库，删了自动重建。
    """
    tag = f"k1-{k1:g}_b-{b:g}"
    return INDEX_DIR / f"bm25_{tag}.pkl", INDEX_DIR / f"bm25_{tag}.meta.json"


def fingerprint(path: Path) -> dict:
    """缓存有效性的判据：语料变了就重建。k1/b 已体现在文件名里，不必再比。"""
    st = path.stat()
    return {
        "corpus": path.name,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def get_index(rows: list[dict], corpus_path: Path, k1: float, b: float,
              use_cache: bool = True, rebuild: bool = False) -> tuple[BM25Index, float, bool]:
    """返回 (索引, 耗时ms, 是否来自缓存)。use_cache=False 时只建不落盘（--site 用）。"""
    pkl, meta_path = index_paths(k1, b)
    fp = fingerprint(corpus_path)

    if use_cache and not rebuild and pkl.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:      # 写一半被中断之类，当作没有缓存
            meta = {}
        if (all(meta.get(k) == v for k, v in fp.items())
                and meta.get("docs") == len(rows)):
            t0 = time.perf_counter()
            idx = BM25Index.load(pkl)
            return idx, (time.perf_counter() - t0) * 1000, True

    t0 = time.perf_counter()
    idx = BM25Index(k1=k1, b=b).build((r["chunk_id"], r["text"]) for r in rows)
    elapsed = (time.perf_counter() - t0) * 1000

    if use_cache:
        idx.save(pkl)
        meta = {**fp, "k1": k1, "b": b, "docs": len(rows), "terms": len(idx.postings),
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return idx, elapsed, False


# ---------- 输出 ----------

def show(query: str, idx: BM25Index, by_id: dict[str, dict], k: int,
         tokens: bool = False) -> None:
    """一条查询：分数、chunk_id、标题路径、URL、正文摘要。

    检索阶段就把出处打出来，是因为等到生成阶段再回头找「这句引用来自哪」，
    十有八九找不到 —— 那时手里只剩一段文本，没有锚点。
    """
    if tokens:
        print(f"\n  分词：{tokenize(query)}")

    t0 = time.perf_counter()
    hits = idx.search(query, k=k)
    ms = (time.perf_counter() - t0) * 1000

    print(f"\n查询：{query}")
    print(f"  （{ms:.1f} ms，命中 {len(hits)} 条）")
    if not hits:
        print("  无命中 —— 先跑 :tok 看分词结果，再决定是加同义词还是改切分规则。")
        return

    for rank, (chunk_id, score) in enumerate(hits, 1):
        r = by_id.get(chunk_id, {})
        print(f"  [{rank}] {score:.3f}  {chunk_id}")
        print(f"       {r.get('title_path', '')}")
        print(f"       {r.get('url', '')}")
        print(f"       {snippet(r.get('text', ''))}")


def repl(idx: BM25Index, by_id: dict[str, dict], k: int, tokens: bool) -> None:
    """交互检索。输入查询回车就查；:k 5 改条数，:tok 开关分词预览，:q 退出。"""
    print("\n" + "=" * 58)
    print("输入查询回车检索    :k 5 改返回条数    :tok 开关分词预览    :q 退出")
    print("=" * 58)

    while True:
        try:
            raw = input("\n查询> ").strip()
        except (EOFError, KeyboardInterrupt):   # 管道输入结束 / Ctrl+C 都算正常退出
            print()
            return
        if not raw:
            continue
        if raw in (":q", "quit", "exit", "q"):
            return
        if raw == ":tok":
            tokens = not tokens
            print(f"分词预览 {'开' if tokens else '关'}")
            continue
        if raw.startswith(":k"):
            parts = raw.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit() and int(parts[1]) > 0:
                k = int(parts[1])
                print(f"返回条数改为 {k}")
            else:
                print("用法：:k 5")
            continue

        show(raw, idx, by_id, k, tokens=tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description="BM25 检索（不带参数＝示例 + 交互模式）")
    ap.add_argument("query", nargs="?", default=None,
                    help="查询语句。留空则先跑示例，再进交互模式")
    ap.add_argument("-k", type=int, default=5, help="返回前 k 条，默认 5")
    ap.add_argument("--site", default=None, help="只在这个域名内检索（会跳过索引缓存）")
    ap.add_argument("--corpus", default=str(CORPUS), help="语料路径")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--rebuild", action="store_true", help="忽略缓存，强制重建索引")
    ap.add_argument("--show-tokens", action="store_true", help="打印查询的分词结果")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        sys.exit(f"语料不存在：{corpus_path}\n先跑一次 ingest：python scripts/build.py --skip-crawl")

    rows = load_corpus(corpus_path)
    if not rows:
        sys.exit(f"语料是空的：{corpus_path}")

    # --site 是把文档整批排除在索引外（BM25 本身没有过滤能力），
    # 这样建出来的索引只对这一个域名有效，绝不能覆盖全量缓存。
    site = args.site
    if site:
        rows = [r for r in rows if r["site"] == site]
        if not rows:
            sys.exit(f"--site {site} 没有命中任何文档，可用域名见 data/reports/ingest_latest.json")

    idx, ms, cached = get_index(rows, corpus_path, args.k1, args.b,
                                use_cache=not site, rebuild=args.rebuild)

    scope = f"（仅 {site}）" if site else ""
    print(f"语料 {len(rows)} 篇{scope} / 词表 {len(idx.postings)} / "
          f"平均长度 {idx.avgdl:.1f} 词 / k1={idx.k1} b={idx.b}")
    if cached:
        print(f"索引从缓存加载，{ms:.0f} ms")
    elif site:
        print(f"索引已构建（--site 是子集索引，不落缓存），{ms:.0f} ms")
    else:
        print(f"索引已构建并写入缓存，{ms:.0f} ms")

    by_id = {r["chunk_id"]: r for r in rows}

    # jieba 词典是首次分词时才加载的（约 0.4 s）。不预热的话，这个成本会记在
    # 「第一条查询的耗时」上，而那条数字是要拿去看 p95 的 —— 会把 jieba 的锅算到 BM25 头上。
    tokenize("预热")

    # 带参数：一条查询跑完就退出，方便在脚本里用
    if args.query:
        show(args.query, idx, by_id, args.k, tokens=args.show_tokens)
        return

    for q in SAMPLE_QUERIES:
        show(q, idx, by_id, k=min(args.k, 3), tokens=args.show_tokens)

    repl(idx, by_id, args.k, args.show_tokens)


if __name__ == "__main__":
    main()
