"""语料抓取：只负责把 HTML 原样存到 data/raw/，不做任何清洗。

清洗是 parse.py 的事。两段分开跑，原因：清洗规则一定会改很多遍，
分离之后改规则只需重跑解析，不用重爬。

用法（PowerShell，项目根目录下）：
    python -m src.ingest.crawl --discover     # 只探测目录页，输出链接数，写 data/urls.txt
    python -m src.ingest.crawl --limit 5      # 抓前 5 篇（验收用）
    python -m src.ingest.crawl                # 全量抓，已存在的自动跳过（断点续爬）

跑之前必须改的两处：CATALOG_URL 和 LINK_SELECTOR（标了 TODO）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
URLS_FILE = ROOT / "data" / "urls.txt"

# TODO 1：填目标站的「目录页 / 索引页」URL，也就是一页能列出所有章节链接的那个页面
CATALOG_URL = ""

# TODO 2：打开目录页源码（浏览器 F12 → Elements），找到包住章节链接的那个容器的
# CSS 选择器。常见形态：".sidebar a"、"nav a"、"#toc a"、"article a"。
# 先用 Python 试，别靠猜。
LINK_SELECTOR = ""

# 只保留同域名链接。跨域名的一般是外链、广告、友情链接
SAME_DOMAIN_ONLY = True

# 只抓某个路径前缀（比如 "/docs/"）就填进来；留空表示不限制
PATH_PREFIX = ""

UA = "rag-hybrid-retrieval-qa/0.1 (learning project; contact: 2596468147@qq.com)"


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def discover() -> list[str]:
    """抓目录页 → 抽链接 → 去重 → 按 URL 字典序排序。

    字典序排序这一步不能省：doc_id 按这个顺序编号，编号稳定，
    以后 C++ 版和 Python 版做 top-10 比对才能对齐。
    """
    if not CATALOG_URL or not LINK_SELECTOR:
        sys.exit("先把 CATALOG_URL 和 LINK_SELECTOR 填上")

    resp = session().get(CATALOG_URL, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    base_domain = urlparse(CATALOG_URL).netloc
    links: set[str] = set()
    for a in soup.select(LINK_SELECTOR):
        href = a.get("href")
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        url = urljoin(CATALOG_URL, href).split("#")[0].rstrip("/")
        if SAME_DOMAIN_ONLY and urlparse(url).netloc != base_domain:
            continue
        if PATH_PREFIX and PATH_PREFIX not in urlparse(url).path:
            continue
        links.add(url)

    urls = sorted(links)
    URLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    URLS_FILE.write_text("\n".join(urls) + "\n", encoding="utf-8")

    print(f"探测到 {len(urls)} 个章节链接")
    print("前 3 条样例：")
    for u in urls[:3]:
        print("  " + u)
    if len(urls) < 800:
        print(
            "\n[警告] 不到 800 篇。要么选择器不对（只抽到导航栏），"
            "要么这个站撑不起 3000 篇，考虑换站。"
        )
    return urls


def read_urls() -> list[str]:
    if not URLS_FILE.exists():
        sys.exit("data/urls.txt 不存在，先跑 --discover")
    return [ln.strip() for ln in URLS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]


def fetch_all(limit: int | None = None) -> None:
    urls = read_urls()
    if limit:
        urls = urls[:limit]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    sess = session()
    got, skipped, failed = 0, 0, []

    for i, url in enumerate(urls, start=1):
        doc_id = f"{i:04d}"
        out = RAW_DIR / f"{doc_id}.html"

        if out.exists() and out.stat().st_size > 0:
            skipped += 1
            continue

        try:
            resp = sess.get(url, timeout=20)
            resp.raise_for_status()
        except Exception as e:  # 单个失败不该中断整批
            failed.append((doc_id, url, repr(e)))
            print(f"[FAIL] {doc_id} {url} -> {e!r}")
            continue

        # 存字节，不存 str。编码问题留给 parse.py 处理，raw 保持原样
        out.write_bytes(resp.content)
        size = len(resp.content)
        flag = "  <-- 偏小，疑似反爬页/JS空壳" if size < 20000 else ""
        print(f"[OK] {doc_id} {size // 1024} KB{flag}")
        got += 1
        time.sleep(0.8)

    print(f"\n完成：新抓 {got} 篇，跳过 {skipped} 篇，失败 {len(failed)} 篇")
    if failed:
        print("失败清单（前 5 条）：")
        for item in failed[:5]:
            print("  ", item)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true", help="只探测目录页并写 urls.txt")
    ap.add_argument("--limit", type=int, default=None, help="只抓前 N 篇，验收用")
    args = ap.parse_args()

    if args.discover:
        discover()
    else:
        fetch_all(args.limit)


if __name__ == "__main__":
    main()
