"""Cross-Encoder 重排：让「评分」和「排序」由同一个模型来做。

为什么要有这一层：
    RRF 的 Top-50 已经覆盖了 20/20 的 gold，召回到顶了。剩下 6 条
    （q003/q005/q008/q010/q012/q020）全在池子里，只是排在第 7~53 名 ——
    瓶颈是「像不像」的判据太粗，不是候选不够。
    bi-encoder 把 query 和 chunk 各自压成一个点再算余弦，信息在压缩时就丢了；
    cross-encoder 把两者拼成一条序列进注意力，能逐词对齐，判得准得多。
    代价是不能预计算：每个 query 都要现场算 N 对，所以只能当决赛评委。

为什么重排 50 条而不是 100 条：
    池子再大，边际收益递减、延迟线性上涨。先用 depth=50 把数字拿出来；
    如果 Recall 卡在 0.75~0.8 上不去，再试 100，看是不是「前 50 里压根没有正确答案」。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
MODEL_NAME = "BAAI/bge-reranker-base"
MAX_LENGTH = 512          # 中位 chunk 346 字；34% 超过 500 字，尾部会被截
BATCH_SIZE = 50           # 正好一次吃完 50 对，实测 47ms（RTX 4060）

# 和 vector.py 同一套：缓存位置不一致会重新下一遍到用户目录。必须在 import 之前设。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

def load_texts(corpus_path: Path = CORPUS_PATH, with_title: bool = True) -> dict[str, str]:
    """chunk_id -> 送进重排器的正文。重排要原文，向量表里只有数。

    为什么默认把 title_path 拼在前面：
        语料是按标题层级切的，同一篇文章里相邻两节的正文往往高度相似，
        只给正文，重排器分不清「两节都提到 token，哪节才是回答这个问题的」。
        标题是免费的上下文——不用改语料、不加 token 成本（中位 133 token，离 512 还远）。
        实测（20 条）：纯正文 0.5667 → 末级标题 0.5583 → 完整路径 0.5833，
        只加末级反而更差，因为丢了「这是哪篇文章的哪一节」这层定位信息。
        with_title=False 留给 A/B，别删。
    """
    texts: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = r["text"]
            title = r.get("title_path") or ""
            if with_title and title:
                text = f"{title}\n{text}"
            texts[r["chunk_id"]] = text
    return texts


class Reranker:
    """包装一个初筛检索器，对外仍是 search(query, k) —— 所以还能被再包一层。"""

    def __init__(self, base, texts: dict[str, str], model_name: str = MODEL_NAME,
                 depth: int = 50, max_length: int = MAX_LENGTH,
                 batch_size: int = BATCH_SIZE, device: str | None = None):
        self.base = base
        self.texts = texts
        self.model_name = model_name
        self.depth = depth
        self.max_length = max_length
        self.batch_size = batch_size
        self._device = device
        self._model = None            # 懒加载：只跑 BM25 的时候不必付这 12 秒

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            # 默认套 Sigmoid：只压缩数值、不改变顺序，排序结果不受影响。
            # 想看 -10~+2 的原始 logit（调参时直观得多），加 activation_fn=torch.nn.Identity()
            self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=device)
            # fp16 只影响速度，不影响排序：实测指标逐位一致（0.5667/0.5292），
            # 延迟 588ms → 198ms。CPU 上 half() 会报错，所以只在 cuda 时启用。
            if device == "cuda":
                self._model.model.half()
        return self._model

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        d = max(k, self.depth)        # 同 RRFFusion：depth < k 时池子不够，等于自废召回
        cand = self.base.search(query, d)
        if not cand:
            return []

        # 正文缺失就跳过，不抛异常：语料重切之后 id 可能短暂对不上，
        pairs: list[tuple[str, str]] = []
        cids: list[str] = []
        for cid, _ in cand:
            text = self.texts.get(cid)
            if text:
                pairs.append((query, text))
                cids.append(cid)
        if not pairs:
            return []

        scores = np.asarray(self.model.predict(pairs, batch_size=self.batch_size))
        order = np.argsort(-scores)[:k]
        return [(cids[i], float(scores[i])) for i in order]



