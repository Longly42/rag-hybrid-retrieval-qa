"""环境自检：向量检索这一路能不能开工，跑这一个脚本就知道。

四道关，缺一条都说明环境没立起来：

    1. 设备  —— torch 认不认 GPU，显存够不够（不够的话后面编码 16k chunk 会 OOM）
    2. 形状  —— bge-base-zh-v1.5 的输出必须是 768 维，且和语料行数无关
    3. 自洽  —— 同一句话编码两次，余弦必须 ≈ 1.0
    4. 语义  —— 意思相近的两句要明显高于不相关的第三句

为什么不是「能跑通就行」：
    装完 torch + 模型，最坏的情况不是报错，而是**静默错**。
    模型加载成了别的维度、忘了 mean pooling、encode 没归一化、把 last_hidden_state
    的首 token 当句向量 —— 这些都不会抛异常，跑起来一样出结果，只是相似度全是噪声。
    等你在 16k chunk 上建完索引、评测数字不对，再回头怀疑到这一层，成本高得多。
    所以这里用四个能一票否决的硬指标把它钉死。

    第 3 条尤其要紧：同一句话两次编码余弦不是 1.0，说明向量没归一化，
    或者模型没有正确进入 eval 模式（dropout 还在随机丢）。归一化没做，
    「内积即余弦」这个前提就不成立，后面暴力检索的分数全是错的量纲。

为什么第 4 条用项目里的句子而不是 Hello World：
    用「今天天气不错」验语义，只能证明模型会说话，证明不了它在你的语料上有区分度。
    这里挑的三句，两句是评测集 q011 的同一个坑（二维列表改一行全变），
    第三句是 Django 静态文件 —— 同域 vs 跨域，正好模拟 symptom 类查询的真实难度。

用法（项目根目录，venv 解释器）：
    ./venv/Scripts/python.exe scripts/check_env.py
    ./venv/Scripts/python.exe scripts/check_env.py --model BAAI/bge-base-zh-v1.5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 模型下载走镜像，且落在项目内，不污染用户目录的全局 HF 缓存：
#   - 国内直连 huggingface.co 基本拉不动，hf-mirror 是官方推荐的镜像
#   - 放 models/ 下面，删项目就等于删干净，也和 .gitignore 里的 models/ 对应
# 必须在 import sentence_transformers 之前设置 —— huggingface_hub 在 import 时
# 就把缓存路径读进常量了，之后再改环境变量不生效。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

PASS, FAIL = "  [PASS]", "  [FAIL]"

# 768 是 bge-base 的硬指标。写成常量而不是从模型里读出来再自己比自己，
# 否则「模型输出 1024 维」也能自洽通过 —— 那多半是加载成了 bge-large。
EXPECT_DIM = 768

SENT_A = "用中括号 0 乘 3 再乘 5 建了个二维列表，改其中一行，结果五行全都变了"
SENT_B = "列表乘法只是复制了元素的引用，五个位置指向同一个列表对象"
SENT_C = "Django 部署到服务器以后，静态文件路径要交给 Web 服务器处理"


def cos(a, b) -> float:
    """归一化向量的点积就是余弦。没归一化的话这里会算错，所以第 3 关先验归一化。"""
    return float(sum(x * y for x, y in zip(a, b)))


def main() -> None:
    ap = argparse.ArgumentParser(description="向量检索环境自检")
    ap.add_argument("--model", default="BAAI/bge-base-zh-v1.5")
    args = ap.parse_args()

    failures: list[str] = []

    # ---------- 依赖 ----------
    print("=" * 60)
    print("依赖版本")
    print("=" * 60)
    try:
        import numpy as np
        import torch
        import sentence_transformers as st
    except ImportError as e:
        sys.exit(f"依赖没装全：{e}\n先跑 pip install torch sentence-transformers")

    print(f"  python               {sys.version.split()[0]}")
    print(f"  torch                {torch.__version__}")
    print(f"  sentence-transformers {st.__version__}")
    print(f"  numpy                {np.__version__}")

    # ---------- 第 1 关：设备 ----------
    print("\n" + "=" * 60)
    print("第 1 关：设备")
    print("=" * 60)
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  CUDA 可用，设备 {name}，显存 {total:.0f} MiB")
        print(f"  torch 编译的 CUDA 版本 {torch.version.cuda}")
        device = "cuda"
    else:
        # CPU 也能跑，只是编码 16k chunk 会从几分钟变成十几分钟。
        # 不判失败，但要明确告诉人「慢在哪」，别到时候误以为是代码写错了。
        print("  CUDA 不可用，回退 CPU（能跑，但编码 16k chunk 会明显变慢）")
        device = "cpu"

    # ---------- 加载模型 ----------
    print("\n" + "=" * 60)
    print(f"加载模型 {args.model}")
    print(f"缓存目录 {os.environ['HF_HOME']}")
    print("=" * 60)
    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(args.model, device=device)
    except Exception as e:
        sys.exit(f"模型加载失败：{type(e).__name__}: {e}\n"
                 "  常见原因：网络拉不动（确认 HF_ENDPOINT 已生效）、磁盘不够、模型名拼错")
    print(f"  加载耗时 {time.perf_counter() - t0:.1f}s")
    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量 {params / 1e6:.1f}M")
    print(f"  max_seq_length {model.max_seq_length}")

    # ---------- 第 2 关：形状 ----------
    print("\n" + "=" * 60)
    print("第 2 关：输出形状")
    print("=" * 60)
    one = model.encode(["测试"], normalize_embeddings=True, convert_to_numpy=True)
    dim = one.shape[-1]
    print(f"  输入 1 句 → 输出 {one.shape}")
    if one.ndim == 2 and dim == EXPECT_DIM:
        print(PASS + f" 维度 {dim}，与 bge-base 预期一致")
    else:
        failures.append(f"输出形状 {one.shape}，预期 (N, {EXPECT_DIM})")
        print(FAIL + f" 预期 (N, {EXPECT_DIM})")

    # ---------- 第 3 关：自洽 ----------
    print("\n" + "=" * 60)
    print("第 3 关：同一句话两次编码")
    print("=" * 60)
    e1 = model.encode([SENT_A], normalize_embeddings=True, convert_to_numpy=True)[0]
    e2 = model.encode([SENT_A], normalize_embeddings=True, convert_to_numpy=True)[0]
    self_cos = cos(e1, e2)
    norm1 = float(np.linalg.norm(e1))
    print(f"  两次编码余弦 {self_cos:.6f}   向量模长 {norm1:.6f}")
    # 1e-4 是给浮点误差留的余量，不是给模型随机性留的余量：
    # 真出现 0.999 这种，说明 dropout 没关或者没归一化，不能放过。
    if abs(self_cos - 1.0) < 1e-4:
        print(PASS + " 完全一致，且已归一化（模长 = 1）")
    else:
        failures.append(f"同句余弦 {self_cos:.6f}，模长 {norm1:.4f}")
        print(FAIL + " 不一致：检查是否 normalize_embeddings=True、模型是否 eval 模式")

    # ---------- 第 4 关：语义区分 ----------
    print("\n" + "=" * 60)
    print("第 4 关：语义区分度")
    print("=" * 60)
    ea, eb, ec = model.encode([SENT_A, SENT_B, SENT_C],
                              normalize_embeddings=True, convert_to_numpy=True)
    sim_ab, sim_ac = cos(ea, eb), cos(ea, ec)
    print(f"  A 与 B（同义，列表引用坑）    {sim_ab:.4f}")
    print(f"  A 与 C（无关，Django 静态文件）{sim_ac:.4f}")
    print(f"  差值 {sim_ab - sim_ac:+.4f}")
    # 门槛定 0.10，是实测之后定的：bge-zh 在中文上的余弦整体偏高，无关句也能到 0.43 左右，
    # 这两句实测差值 0.13 —— 拿英文模型的经验（动辄 0.3+）来设 0.3 的门槛，
    # 会把加载正确的模型判成失败。0.10 是「确实分得开」的下限，再低就只剩噪声了。
    if sim_ab - sim_ac > 0.10:
        print(PASS + " 同义明显高于无关，模型加载正确")
    else:
        failures.append(f"语义差值只有 {sim_ab - sim_ac:+.4f}")
        print(FAIL + " 区分度不足：检查模型名、是否加载错了维度")

    # ---------- 参考：吞吐 ----------
    # 不是验收项，只是给第 2 步（编码 16,378 chunk）估时间用。
    print("\n" + "=" * 60)
    print("参考：编码吞吐（用来估第 2 步要跑多久）")
    print("=" * 60)
    batch = [SENT_A, SENT_B, SENT_C] * 11     # 33 条，够 warmup 一次
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.encode(batch, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  33 条 {dt * 1000:.0f} ms（含首次 warmup，偏保守）")
    print(f"  按此推算 16,378 条约 {16378 / len(batch) * dt / 60:.1f} 分钟（实际会更快）")

    # ---------- 显存 ----------
    if device == "cuda":
        alloc = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.max_memory_reserved() / 1024**2
        print(f"\n  峰值显存 已分配 {alloc:.0f} MiB / 保留 {reserved:.0f} MiB")
        if reserved > 7000:
            failures.append(f"峰值显存 {reserved:.0f} MiB，8G 卡余量太小")
            print(FAIL + " 8G 卡余量太小，第 2 步要调小 batch_size 或用 fp16")
        else:
            print(PASS + " 显存余量充足")

    # ---------- 结论 ----------
    print("\n" + "=" * 60)
    if failures:
        print(f"未通过（{len(failures)} 项）：")
        for f in failures:
            print(f"  - {f}")
        print("=" * 60)
        sys.exit(1)
    print("四条全过，向量检索这一路可以开工了。")
    print("=" * 60)


if __name__ == "__main__":
    main()
