"""一次性迁移：把 data/raw 下扁平的 NNNN.html 搬进「格式 / 域 / 源路径」三层结构。

为什么必须迁：
    文件名是 urls.txt 里的**位置编号**，重爬一次就可能整批错位。这个坑已经踩过两次
    （0001-0016 全是不属于 library 的旧教程页），而且它一声不响。
    路径来自页面自己的 <link rel="canonical">，页面是谁由页面自己声明，重爬多少次都不会变。

迁移规则：
    data/raw/0004.html
      → data/raw/html/docs.python.org/3/tutorial/controlflow.html

    第一层是格式（html / rst / md），第二层是域名，第三层起原样保留 canonical 的路径。
    以后加 Django 的 rst 源就是 data/raw/rst/docs.djangoproject.com/zh-hans/5.2/topics/db/models.txt，
    同一个频道的同一篇文档天然挨在一起，按站筛选靠目录就够了。

无 canonical 的文件：
    页面没声明身份，就没法知道它是谁。统一丢进 data/raw/html/_nosource/，
    保持原名。这类文件 parse.py 会照样跳过并报「canonical 缺失」，留着是为了
    将来能人工核对，不是为了进语料。

幂等：
    只扫 data/raw/ 的第一层（glob 不递归）。迁完之后文件都在子目录里，
    再跑一次什么也不做，可以放心重复执行。

用法：
    python scripts/migrate_raw_layout.py --dry-run    # 只打印计划，不动文件
    python scripts/migrate_raw_layout.py              # 实际搬
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# 无 canonical 时的兜底目录名。下划线开头，一眼能看出不是正常站点。
NOSOURCE = "_nosource"


def canonical_of(path: Path) -> str:
    """读一个 HTML 文件，返回它自己声明的 canonical / og:url，没有就返回空串。

    先 canonical 再 og:url：canonical 是搜索引擎认定的规范地址，
    og:url 只是社交分享用的，优先级更低，但总比没有强。
    """
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    tag = soup.find("link", attrs={"rel": ["canonical"]})
    if tag and tag.get("href"):
        return tag["href"].strip()
    meta = soup.find("meta", attrs={"property": "og:url"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    return ""


def target_path(url: str, fallback_name: str) -> Path:
    """把 URL 换算成 data/raw 下的相对路径。

    https://docs.python.org/3/library/bisect.html
      → html/docs.python.org/3/library/bisect.html

    路径段里的 Windows 非法字符（: * ? " < > |）替换成下划线。
    URL 里带查询串的一律丢掉 —— 存下来的字节和查询串无关。
    """
    if not url:
        return Path("html") / NOSOURCE / fallback_name

    p = urlparse(url)
    site = p.netloc or NOSOURCE

    segs = [s for s in p.path.split("/") if s]
    if not segs:
        # 站点根页面：example.com/ 这种，用 index.html 占位，否则路径会退化成目录
        segs = ["index.html"]

    # 路径段合法化。域名里的 : 只在带端口时出现，一并处理
    safe_segs = [s.replace(":", "_").replace("*", "_").replace("?", "_")
                  .replace('"', "_").replace("<", "_").replace(">", "_")
                  .replace("|", "_") for s in segs]
    site = site.replace(":", "_")

    # 最后一段没有扩展名（Django 那种目录式 URL）就补 .txt，
    # 否则 Path 会把它当目录，文件落地变成一个空目录名
    if "." not in safe_segs[-1]:
        safe_segs[-1] = safe_segs[-1] + ".html"

    return Path("html") / site / Path(*safe_segs)


def digest(path: Path) -> str:
    """文件内容的 sha1，用来判断「目标已存在时，是不是同一份内容」。"""
    return hashlib.sha1(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印迁移计划，不动文件")
    args = ap.parse_args()

    # 只扫第一层。这个 glob 不递归，所以迁移过的文件不会在下次运行时被再次选中
    files = sorted(RAW_DIR.glob("*.html"))
    if not files:
        print("data/raw/ 第一层没有 *.html，要么已经迁完，要么路径不对。")
        return

    print(f"待处理 {len(files)} 个文件" + ("（dry-run，不会改动磁盘）" if args.dry_run else ""))

    moved, merged, conflicts, nosource = 0, 0, 0, 0
    plan: list[tuple[Path, Path]] = []

    for src in files:
        url = canonical_of(src)
        rel = target_path(url, src.name)
        if not url:
            nosource += 1
        plan.append((src, RAW_DIR / rel))

    # 先全部算完再动手：万一中途出错，你能从打印出来的计划里看出停在哪
    for src, dst in plan:
        if args.dry_run:
            print(f"  {src.name:>10}  →  {dst.relative_to(RAW_DIR).as_posix()}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            # 同一篇文档抓了两份（重爬造成的）。内容一样就只留一份，
            # 不一样就都留着并报出来 —— 内容不同说明页面变过，值得人看一眼
            if digest(src) == digest(dst):
                src.unlink()
                merged += 1
            else:
                stem, suffix = dst.stem, dst.suffix
                alt = dst.with_name(f"{stem}-dup{suffix}")
                shutil.move(str(src), str(alt))
                conflicts += 1
                print(f"  [冲突] {src.name} 与已有 {dst.name} 内容不同，存为 {alt.name}")
            continue

        shutil.move(str(src), str(dst))
        moved += 1

    if args.dry_run:
        print(f"\n计划：{len(plan)} 个文件，其中 {nosource} 个无 canonical（会进 html/_nosource/）")
        return

    print(f"\n完成：搬移 {moved}，合并重复 {merged}，内容冲突 {conflicts}，无 canonical {nosource}")

    # 迁移后自检：第一层必须已经空了，子目录里的文件数必须等于原来那一批
    left = list(RAW_DIR.glob("*.html"))
    under = list(RAW_DIR.rglob("*.*"))
    print(f"自检：第一层残留 {len(left)} 个（应为 0），子目录共 {len(under)} 个文件")

    if under:
        by_fmt: dict[str, int] = {}
        for f in under:
            top = f.relative_to(RAW_DIR).parts[0]
            by_fmt[top] = by_fmt.get(top, 0) + 1
        print("按格式：" + "，".join(f"{k} {v}" for k, v in sorted(by_fmt.items())))


if __name__ == "__main__":
    main()
