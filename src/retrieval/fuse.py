"""RRF 融合：只看名次，不看分数。

score(d) = Σ_i 1 / (k + rank_i(d))

为什么是 RRF 而不是分数加权：
    实测这张语料上 BM25 的分数无界（0~20+），bge 的余弦全挤在 0.5~0.75、
    Top-1 和 Top-50 只差 0.06。两者放一起加权，权重怎么调都是猜 ——
    而且每换一次语料、每重切一次 chunk 就得重调一遍。
    名次是天然可比的：两路的第一名都是「它认为最像的那条」。

为什么 k=60：
    论文里的经验值。作用是压平头部差距 —— 1/(60+1)=0.0164，1/(60+50)=0.0091，
    第一名只比第五十名重要不到两倍。k 越小越偏向单路的头名，越大越平均。
    20 条评测集上把这个参数调来调去，调出来的多半是噪声，所以固定 60。
"""


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: 每路的 [(chunk_id, score), ...]，已按各自分数降序。

    返回 [(chunk_id, rrf_score)]，按融合分降序 —— 注意分数换了量纲，
    不再是余弦也不是 BM25，只在这份融合名单内部有意义。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, 1):   # 名次从 1 开始
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class RRFFusion:
    """把若干检索器合成一个 —— 对外仍是 search(query, k)，所以还能被再融合。"""

    def __init__(self, retrievers: list, k: int = 60, depth: int = 50):
        if len(retrievers) < 2:
            raise ValueError("融合至少要两路，单路融合没有意义")
        self.retrievers = retrievers
        self.k = k
        self.depth = depth

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        # 深度不够，融合就只能在浅层里挑，等于自废召回：
        # q008/q009 的答案在第 15 名，depth=10 的话它们根本进不了融合池。
        d = max(k, self.depth)
        rankings = [r.search(query, d) for r in self.retrievers]
        return rrf_fuse(rankings, k=self.k)[:k]
