| 日期 | 改动 | Recall@5 | MRR | p95(ms) | 备注 |
| --- | --- | --- | --- | --- | --- |
| 2026-11-20 | 建立工程骨架、抓取脚本骨架 | – | – | – | 语料尚未建立，无基线。README 中的 88% / 260ms / 92% 均为目标值 |
| 2026-11-26 | 建立 20 条评测集（data/eval/qa.jsonl），纯 BM25 基线，k1=1.5 b=0.75，Top-5 | 0.3833 | 0.3392 | 2.7 | 单次查询中位 1.7ms。分组：identifier 0.500 / concept 0.500 / howto 0.417 / symptom 0.167 —— symptom 类是最明确的改进目标 |
| 2026-11-27 | 向量检索环境就位：torch 2.9.1+cu128（RTX 4060，CUDA 12.8 可用）、sentence-transformers 6.3.0、bge-base-zh-v1.5 权重落 models/hf，新增 scripts/check_env.py 四道自检 | – | – | – | 只装环境，检索逻辑未动，指标与 11-26 完全一致（复跑 0.3833 / 0.3392 / p95 2.5ms）。编码吞吐 33 条 89ms（GPU），16,378 chunk 预计 1 分钟内；峰值显存 437MiB |
