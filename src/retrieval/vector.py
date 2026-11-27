"""向量检索内核：一张向量表 + 一次矩阵乘。

为什么是暴力精确检索，不上 faiss / Chroma：
    16,378 × 768 的内积 = 1,200 万次乘加，BLAS 上几毫秒就完了。
    这个规模上任何 ANN 索引都是「为了省 3ms 引入一层依赖 + 一套参数」，
    而且暴力算的结果是精确的 —— 调参阶段关心「召回上限在哪」，
    不关心「近似索引又啃掉了几个点」。
    等语料上十万、延迟真成瓶颈了再换索引，search(query, k) 的签名不用变。

和 BM25 的关系：
    BM25Index.search(query, k)   -> [(chunk_id, score)]
    VectorIndex.search(query, k) -> [(chunk_id, score)]
    签名刻意做成一样 —— 第 4 步的 eval.py 才能用同一套代码跑两路，
    第 5 步的 RRF 才能无脑对两个列表做名次融合。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB_PATH = ROOT / "data" / "index" / "embeddings.npy"
IDS_PATH = ROOT / "data" / "index" / "chunk_ids.json"
MODEL_NAME = "BAAI/bge-base-zh-v1.5"

# 模型缓存位置必须和 embed_corpus.py 一致，否则这里会重新下一遍到用户目录。
# 这套设置现在散在 check_env / embed_corpus / vector 三处，
# 第 4 步重构时抽到 src/config.py 统一。必须在 import sentence_transformers 之前设。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

# bge 官方建议查询加这句指令，对 v1.5 是可选项、提升很小。
# 默认关着：加了之后 query 和 chunk 就不在同一侧，得单独 A/B，别和别的改动混在一起。
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

class VectorIndex:
    """向量表 + 一个懒加载的编码器。"""

    def __init__(self, emb: np.ndarray, ids: list[str], model_name: str, device: str | None = None):
        self.emb = emb                  # (N, 768) float32，已归一化
        self.ids = ids                  # 第 i 个就是 emb 第 i 行
        self.model_name = model_name
        self._device = device
        self._model = None              # 懒加载：只做打分时不必付加载模型的 5 秒

    @classmethod
    def load(cls, emb_path: Path = EMB_PATH, ids_path: Path = IDS_PATH,
             model_name: str = MODEL_NAME, device: str | None = None) -> "VectorIndex":
        # mmap：加载瞬间完成，48MB 不立刻进内存，算的时候由 OS 按页读。
        # 以后多进程（比如批量评测开多核）还能共享同一份页缓存。
        emb = np.load(emb_path, mmap_mode="r")
        ids = json.loads(ids_path.read_text(encoding="utf-8"))

        if emb.shape[0] != len(ids):
            raise ValueError(f"向量表 {emb.shape[0]} 行、id {len(ids)} 条，对不上 —— "
                             "产物坏了，重跑 scripts/embed_corpus.py --rebuild")
        if emb.shape[1] != 768:
            raise ValueError(f"维度 {emb.shape[1]}，预期 768 —— 模型和向量表不是同一个")
        return cls(emb, ids, model_name, device)

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def encode(self, texts: list[str], instruction: str | None = None) -> np.ndarray:
        """查询侧编码。必须归一化，理由见模块开头的第 3 条。

        instruction 只加在 query 上，不加在 chunk 上 —— 这是 bge 训练时的非对称用法。
        """
        if instruction:
            texts = [instruction + t for t in texts]
        return self.model.encode(texts, normalize_embeddings=True,
                                 convert_to_numpy=True, batch_size=32)

    def search_by_vector(self, vec: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        # 一行搞定 16378 个余弦：归一化之后，内积就是余弦。
        # 别写成 for 循环逐个算 —— 那是 16378 次 Python 开销，慢三个数量级。
        scores = np.asarray(self.emb @ vec, dtype=np.float32)
        n = scores.shape[0]

        if k >= n:
            order = np.argsort(-scores)
        else:
            # argpartition 只保证「前 k 个都比其余大」，不保证内部有序，
            # 所以还要对这 k 个再排一次；它的价值在 O(n)，语料大了才显形。
            cand = np.argpartition(-scores, k)[:k]
            order = cand[np.argsort(-scores[cand])]

        return [(self.ids[i], float(scores[i])) for i in order]

    def search(self, query: str, k: int = 5, instruction: str | None = None) -> list[tuple[str, float]]:
        return self.search_by_vector(self.encode([query], instruction=instruction)[0], k)
    







