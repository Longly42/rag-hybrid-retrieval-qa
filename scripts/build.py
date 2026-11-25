"""一条命令把 ingest 跑完：抓取（断点续爬）→ 解析切分 → 统计报告落盘。

为什么需要这个入口：
    ingest 由 crawl 和 parse 两段组成，两段各有各的参数。跑一次要记两条命令、
    记住顺序、记住 parse 得等 crawl 跑完 —— 这种「靠记忆复现」的流程，
    隔两个星期回来就一定会漏一步，然后拿着半份语料去调检索参数。
    这个脚本把顺序固定下来，并且把数字写成报告：数字关掉终端就没了，不算证据。

用法（项目根目录，**用 venv 里的解释器**）：
    python scripts/build.py                    # 全量：抓 + 切 + 报告
    python scripts/build.py --skip-crawl       # 只重切（改了解析规则时用这个）
    python scripts/build.py --limit 20         # 冒烟：只切 20 篇，写 data/_smoke.jsonl
    python scripts/build.py --only python-library django-html   # 只跑部分源
    python scripts/build.py --refresh-crawl    # 重新探测上游目录页，会带上新页面

抓取默认按 data/urls/{源}.txt 重建，不重新探测目录页：
    清单是累积的（分栏目抓、多轮抓都合并在里面），目录页只是「此刻挂着什么」。
    Django 按 ref / topics / howto 分三次抓，重新探测一次只返回当前栏目的几十条，
    清单里另外一百多篇根本不在返回结果里 —— 只靠探测，raw 一删就会少一百多篇，
    而且不报错。所以「复原」必须走清单，「扩充」才走探测，两件事分开。

为什么不提供「抓所有配置好的源」：
    SOURCES 里有 7 个源，但真正进语料的只有下面 CORPUS_SOURCES 这 4 个。
    django-rst / python-tutorial / python-reference / python-c-api 是配置好了
    还没验收的源。一键全抓会让它们混进语料 —— 而 django 的 rst 一旦进来，
    按「rst > md > html」的选优规则会把现有的 html 版本整批换掉，
    语料结构、chunk 数、指标全部变，排查起来像灵异事件。
    要加源就显式改这张表，让它成为一次有意识的决定。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 当前语料的五个来源。加源：往这里加一行，然后跑一次看报告里的 docs/chunks 变化。
# python-tutorial 是项目第一天抓的那 16 篇，当时还没有清单机制，导致这 16 篇
# 一直躺在 raw 里却没有清单 —— 删了 raw 就再也重建不出来。2026-11-24 补了
# data/urls/python-tutorial.txt（探测目录页得到 16 条，与 raw 里的文件一一对上）。
CORPUS_SOURCES = (
    "python-library",
    "python-tutorial",
    "django-html",
    "python100days-md",
    "vue-docs-md",
)

CORPUS = ROOT / "data" / "corpus.jsonl"
REPORT = ROOT / "data" / "reports" / "ingest_latest.json"


def check_env() -> None:
    """提前拦一次：系统 Python 里没装 bs4，报错会出现在子进程里，读起来很绕。"""
    try:
        import bs4  # noqa: F401
        import requests  # noqa: F401
    except ImportError as e:
        sys.exit(
            f"当前解释器缺依赖：{e.name}\n"
            f"  解释器：{sys.executable}\n"
            f"  用 venv 跑：./venv/Scripts/python.exe scripts/build.py（Windows）\n"
            f"             ./venv/bin/python scripts/build.py（macOS/Linux）\n"
            f"  缺的包：pip install -r requirements.txt"
        )


def load_prev() -> dict | None:
    if not REPORT.exists():
        return None
    try:
        return json.loads(REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def crawl(sources: tuple[str, ...], refresh: bool = False) -> list[dict]:
    from src.ingest.crawl import SOURCES, fetch_all

    stats: list[dict] = []
    for name in sources:
        if name not in SOURCES:
            print(f"[跳过] SOURCES 里没有 {name}，检查 src/ingest/crawl.py")
            continue
        print(f"\n=== 抓取 {name} ===")
        started = time.time()
        try:
            st = fetch_all(name, from_urls=not refresh)
        except SystemExit as e:      # fetch_all 在「一个链接都没发现」时直接退出
            print(f"[失败] {name}：{e}")
            stats.append({"source": name, "planned": 0, "got": 0,
                          "skipped": 0, "failed": [["-", str(e)]], "warns": 0})
            continue
        except Exception as e:       # 网络/解析异常不该让整批停在这里
            print(f"[失败] {name}：{e!r}")
            stats.append({"source": name, "planned": 0, "got": 0,
                          "skipped": 0, "failed": [["-", repr(e)]], "warns": 0})
            continue
        st["seconds"] = round(time.time() - started, 1)
        stats.append(st)
    return stats


def parse(limit: int | None, out: Path, report: Path) -> None:
    cmd = [sys.executable, "-m", "src.ingest.parse", "--out", str(out), "--report", str(report)]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"\n=== 解析切分 ===\n{' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(f"parse 失败，退出码 {r.returncode}。上面应该有原因。")


def print_summary(rep: dict, prev: dict | None, crawl_stats: list[dict], report: Path) -> None:
    print("\n" + "=" * 58)
    print("ingest 报告")
    print("=" * 58)
    print(f"扫描文件 {rep['files_scanned']} 篇 → {rep['docs']} 篇进语料 "
          f"→ {rep['chunks']} 个 chunk")
    c = rep["chars"]
    print(f"字符数：平均 {c['avg']}，中位 {c['median']}，p90 {c['p90']}，max {c['max']}")
    print(f"含代码 {rep['code_chunks']} 块（{rep['code_pct']}%），"
          f"短于 120 字 {rep['short_lt120']} 块（{rep['short_pct']}%）")

    print("\n按域名：")
    for site, v in rep["by_site"].items():
        print(f"  {site:<26} {v['docs']:>4} 篇  {v['chunks']:>5} chunk  中位 {v['median']} 字")

    print(f"\n丢弃 {rep['skipped_total']} 篇，按原因：")
    if rep["skipped_by_reason"]:
        for why, n in sorted(rep["skipped_by_reason"].items(), key=lambda kv: -kv[1]):
            print(f"  {n:>3} 篇  {why}")
    else:
        print("  无")

    if crawl_stats:
        print("\n抓取：")
        for st in crawl_stats:
            print(f"  {st['source']:<18} 计划 {st['planned']:>4}  新抓 {st['got']:>4}  "
                  f"跳过 {st['skipped']:>4}  失败 {len(st['failed']):>3}"
                  + (f"  {st['seconds']}s" if "seconds" in st else ""))

    # 可复现自检：同一份 raw 重切，应该得到一样的数。差得明显就是代码或数据动了
    if prev:
        d_docs = rep["docs"] - prev.get("docs", 0)
        d_chunks = rep["chunks"] - prev.get("chunks", 0)
        same = (d_docs == 0 and d_chunks == 0)
        old_hash, new_hash = prev.get("corpus_sha256"), rep.get("corpus_sha256")
        print(f"\n与上次报告对比：{'一致' if same else '有变化'}（篇数 {d_docs:+d}，chunk {d_chunks:+d}）")
        if old_hash and new_hash:
            print(f"  语料指纹：{'逐字节一致' if old_hash == new_hash else '内容有差异'}"
                  f"（{new_hash[:12]}）")
        if not same:
            print(f"  上次：{prev.get('generated_at', '?')}  "
                  f"{prev.get('docs', 0)} 篇 / {prev.get('chunks', 0)} chunk")
    try:
        shown = report.relative_to(ROOT).as_posix()
    except ValueError:      # 报告被指到项目外面去了，照原样显示
        shown = str(report)
    print(f"\n报告已写入：{shown}")
    print("=" * 58)


def main() -> None:
    ap = argparse.ArgumentParser(description="ingest 一键流程：抓取 → 切分 → 报告")
    ap.add_argument("--skip-crawl", action="store_true", help="不抓取，只重新切分")
    ap.add_argument("--limit", type=int, default=None,
                    help="只切前 N 篇（冒烟用）。会自动改写到 data/_smoke.jsonl，不碰正式语料")
    ap.add_argument("--out", default=None, help="语料输出路径，默认 data/corpus.jsonl")
    ap.add_argument("--report", default=None, help="报告输出路径，默认 data/reports/ingest_latest.json")
    ap.add_argument("--only", nargs="+", default=None,
                    help="只抓这几个源，默认 CORPUS_SOURCES 里那五个")
    ap.add_argument("--refresh-crawl", action="store_true",
                    help="重新探测目录页（会带上上游新页面，语料会变大）；默认按 data/urls 清单重建")
    args = ap.parse_args()

    check_env()
    # 子进程和父进程共用同一个 stdout，父进程的输出被缓冲就会和 parse 的输出乱序
    sys.stdout.reconfigure(line_buffering=True)

    # 冒烟模式不许写正式语料：--limit 出来的半份语料一旦覆盖 corpus.jsonl，
    # 后面所有评测基线都建在残次品上，而且没人会发现少了东西。
    out = Path(args.out) if args.out else (ROOT / "data" / ("_smoke.jsonl" if args.limit else CORPUS.name))
    report = Path(args.report) if args.report else (
        ROOT / "data" / "_smoke_report.json" if args.limit else REPORT
    )
    # 相对路径按项目根算：不然在别的目录里调用，文件会落到意料之外的地方
    out = out if out.is_absolute() else ROOT / out
    report = report if report.is_absolute() else ROOT / report
    if args.limit:
        print(f"[冒烟模式] 只切前 {args.limit} 篇，输出 {out.relative_to(ROOT).as_posix()}，"
              f"不会覆盖 {CORPUS.name}")

    prev = None if args.limit else load_prev()
    started = time.time()

    crawl_stats: list[dict] = []
    if not args.skip_crawl:
        crawl_stats = crawl(tuple(args.only) if args.only else CORPUS_SOURCES,
                            refresh=args.refresh_crawl)

    parse(args.limit, out, report)

    if report.exists():
        rep = json.loads(report.read_text(encoding="utf-8"))
        print_summary(rep, prev, crawl_stats, report)
    else:
        sys.exit(f"parse 没写出报告：{report}")

    print(f"\n总耗时 {round(time.time() - started, 1)}s")


if __name__ == "__main__":
    main()
