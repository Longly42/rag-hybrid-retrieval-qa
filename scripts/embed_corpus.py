"""编码语料：16,378 个 chunk → 一张 (N, 768) 的向量表。

    python scripts/embed_corpus.py            # 建表；已是最新就跳过
    python scripts/embed_corpus.py --rebuild  # 强制重建
    python scripts/embed_corpus.py --verify   # 只验已落盘的产物，不重新编码

产出（与 BM25 缓存同级，都在 data/index/ 下，不入库）：
    embeddings.npy        float32 (N, 768)，每行已归一化 —— 内积即余弦
    chunk_ids.json        第 i 个 id 就是 embeddings 第 i 行，行序即唯一真相
    embedding_meta.json   模型 / 维度 / 行数 / 语料指纹 / 耗时
"""

from __future__ import annotations #  # 允许在类内使用类名

import argparse #用于命令行选项、参数和子命令解析的模块
import json     #json模块用于处理JSON数据
import os        #os模块提供了许多与操作系统交互的功能
import time      #time模块提供了各种与时间相关的函数
import sys       #sys模块提供了对Python解释器使用或维护的一些变量的访问
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# 必须在 import sentence_transformers 之前设置：huggingface_hub 在 import 时
# 就把缓存路径读进常量了，之后再改环境变量不生效。

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

#定义路径常量
CORPUS = ROOT / "data" / "corpus.jsonl"
OUT_DIR = ROOT / "data" / "index"
EMB = OUT_DIR / "embeddings.npy"
IDS = OUT_DIR / "chunk_ids.json"
META = OUT_DIR / "embedding_meta.json"


def load_corpus(path: Path) -> list[dict]:
    """原样读，不排序不去重 —— 这一行的顺序待会儿就是向量表的行序。"""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def fingerprint(path: Path) -> dict:
    """和 BM25 缓存同一套判据：语料换了就重建。

    只看「文件是否存在」是最容易出灵异事件的做法 —— 语料重切过、向量还是旧的，
    检索结果看着就是不对劲，却查不出原因。
    """
    st = path.stat()
    return {"corpus": path.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}

def is_fresh(rows: list[dict], corpus_path: Path, model_name: str) -> bool:
    """检查缓存是否有效。"""
    if not (EMB.exists() and IDS.exists() and META.exists()):
        return False
    try:
        meta = json.loads(META.read_text(encoding="utf-8"))
    except json.JSONDecodeError:          # 写一半被中断，当没有缓存
        return False
    fp = fingerprint(corpus_path)
    return (all(meta.get(k) == v for k, v in fp.items())
            and meta.get("rows") == len(rows)
            and meta.get("model") == model_name)

def build(rows: list[dict], model_name: str, batch: int, corpus_path: Path) -> None:
    """编码语料"""    
    texts = [r["text"] for r in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备 {device} / 语料 {len(texts)} 条 / 模型 {model_name}")

    model = SentenceTransformer(model_name, device=device)
    t0 = time.perf_counter()
    emb = model.encode(
        texts,
        batch_size=batch,
        normalize_embeddings=True,   # 落盘就是单位向量，检索端不用再归一化
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    emb = np.ascontiguousarray(emb, dtype=np.float32)   # 防御：确保是 fp32、连续内存
    dt = time.perf_counter() - t0

    expect = (len(texts), 768)
    if emb.shape != expect:
        sys.exit(f"形状不对：{emb.shape}，预期 {expect}")
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        sys.exit(f"不是单位向量（模长 {norms.min():.4f}~{norms.max():.4f}），"
                 "检查 normalize_embeddings=True")

    ids = [r["chunk_id"] for r in rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB, emb)
    IDS.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    META.write_text(json.dumps({
        **fingerprint(corpus_path),
        "model": model_name, "dim": int(emb.shape[1]), "rows": len(ids),
        "dtype": str(emb.dtype), "normalized": True, "batch": batch,
        "seconds": round(dt, 1), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"编码完成 {dt:.1f}s，{len(ids)} 条 × {emb.shape[1]} 维")
    print(f"  {EMB.name}      {EMB.stat().st_size / 1024**2:.1f} MB")
    print(f"  {IDS.name}  {IDS.stat().st_size / 1024:.0f} KB")
    if device == "cuda":
        print(f"  峰值显存 {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")

def verify(rows: list[dict], model_name: str | None = None) -> bool:
    """验落盘的产物，不是验内存里的对象。

    内存里 emb 和 ids 当然对得上 —— 它们是同一个循环产出的。真正会出错的地方是
    写盘之后谁跟谁错位了，所以这里一律从磁盘重新读一遍再比。
    """
    emb = np.load(EMB, mmap_mode="r")            # mmap：不真读 48MB 进内存
    ids = json.loads(IDS.read_text(encoding="utf-8"))
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  [PASS] " if cond else "  [FAIL] ") + msg)
        ok = ok and cond

    print(f"行数：语料 {len(rows)} / ids {len(ids)} / 向量 {emb.shape[0]}")
    check(len(rows) == len(ids) == emb.shape[0], "三者行数一致")
    check(emb.shape[1] == 768, f"维度 {emb.shape[1]}")

    by_id = {r["chunk_id"]: r for r in rows}
    # 抽首、中、末三条：错位往往从中间某处开始，只查第 0 行是查不出来的
    for i in (0, len(ids) // 2, len(ids) - 1):
        r = by_id.get(ids[i])
        check(r is not None and r["text"] == rows[i]["text"],
              f"第 {i} 行 ids[{i}] 对得上 corpus 同位置的正文")
        print(f"         {ids[i]}")

    if model_name:
        model = SentenceTransformer(model_name,
                                    device="cuda" if torch.cuda.is_available() else "cpu")
        for i in (0, len(ids) // 2):
            again = model.encode([rows[i]["text"]], normalize_embeddings=True,
                                 convert_to_numpy=True)[0]
            cos = float(np.dot(again, np.asarray(emb[i])))
            check(cos > 0.9999, f"第 {i} 行重新编码余弦 {cos:.6f}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="编码语料为向量表")
    ap.add_argument("--model", default="BAAI/bge-base-zh-v1.5")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--rebuild", action="store_true", help="忽略已有产物，强制重建")
    ap.add_argument("--verify", action="store_true", help="只校验已落盘的产物")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        sys.exit(f"语料不存在：{corpus_path}\n先跑 python scripts/build.py --skip-crawl")

    rows = load_corpus(corpus_path)
    if not rows:
        sys.exit(f"语料是空的：{corpus_path}")

    ids = [r["chunk_id"] for r in rows]
    if len(set(ids)) != len(ids):
        # 重复 id 会让「第 i 行是谁」失去唯一解，检索结果回溯时会串
        sys.exit("chunk_id 有重复，先查语料")

    if args.verify:
        sys.exit(0 if verify(rows, args.model) else 1)

    if not args.rebuild and is_fresh(rows, corpus_path, args.model):
        print("向量表已是最新，跳过（--rebuild 强制重建）")
    else:
        build(rows, args.model, args.batch, corpus_path)

    if not verify(rows, args.model):
        sys.exit("自检未过，产物不可用")


if __name__ == "__main__":
    main()









