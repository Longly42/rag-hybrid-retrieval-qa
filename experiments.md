| 日期 | 改动 | Recall@5 | MRR | p95(ms) | 备注 |
| --- | --- | --- | --- | --- | --- |
| 2026-11-20 | 建立工程骨架、抓取脚本骨架 | – | – | – | 语料尚未建立，无基线。README 中的 88% / 260ms / 92% 均为目标值 |
| 2026-11-26 | 建立 20 条评测集（data/eval/qa.jsonl），纯 BM25 基线，k1=1.5 b=0.75，Top-5 | 0.3833 | 0.3392 | 2.7 | 单次查询中位 1.7ms。分组：identifier 0.500 / concept 0.500 / howto 0.417 / symptom 0.167 —— symptom 类是最明确的改进目标 |
| 2026-11-27 | 向量检索环境就位：torch 2.9.1+cu128（RTX 4060，CUDA 12.8 可用）、sentence-transformers 6.3.0、bge-base-zh-v1.5 权重落 models/hf，新增 scripts/check_env.py 四道自检 | – | – | – | 只装环境，检索逻辑未动，指标与 11-26 完全一致（复跑 0.3833 / 0.3392 / p95 2.5ms）。编码吞吐 33 条 89ms（GPU），16,378 chunk 预计 1 分钟内；峰值显存 437MiB |
| 2026-11-27 | 语料编码落盘：scripts/embed_corpus.py → embeddings.npy（16378×768 fp32，已归一化）+ chunk_ids.json | – | – | – | 编码 156s、峰值显存 985MiB。自检全过：行数/维度/首中末三条 id 对齐/重编码余弦 1.000000 |
| 2026-11-27 | 纯向量检索：src/retrieval/vector.py（归一化内积 + argpartition），暴力精确 Top-5 | 0.4667 | 0.3592 | 24.2 | 命中 12/20，比 BM25 多 2 条。分组：concept 0.600 / howto 0.583 / identifier 0.500 / symptom 0.250（从 0.167 起来）。余弦压缩严重：Top-1 与 Top-50 只差 0.04~0.21，Top-5 边界极挤，所以分数加权融合必死 |
| 2026-11-27 | RRF 融合（src/retrieval/fuse.py，k=60，两路各取 Top-50）+ eval.py 加 --mode bm25/vector/rrf | 0.6000 | 0.4458 | 29.0 | 命中 14/20，正好等于两路 Top-5 并集的上限。分组：concept 0.800 / howto 0.750 / identifier 0.600 / symptom 0.333。代价：howto 的 MRR 从 0.500 退到 0.396（q017 从第 1 掉到第 3），融合买的是召回不是头名精度。**关键诊断：两路并集 Top-50 覆盖 20/20**，说明召回已不是瓶颈，剩下全是排序问题 |

