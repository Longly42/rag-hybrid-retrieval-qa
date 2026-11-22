"""语料抓取：只负责把 HTML 原样存到 data/raw/，不做任何清洗。

清洗是 parse.py 的事。两段分开跑，清洗规则会改很多遍，
分离之后改规则只需重跑解析，不用重爬。

用法（PowerShell，项目根目录下）：
    python -m src.ingest.crawl --discover     # 只探测目录页，输出链接数，写 data/urls.txt
    python -m src.ingest.crawl --limit 5      # 抓前 5 篇（验收用）
    python -m src.ingest.crawl                # 全量抓，已存在的自动跳过（断点续爬）

跑之前必须改的两处：CATALOG_URL 和 LINK_SELECTOR
"""

from __future__ import annotations  #解释这一行：启用未来风格的类型注解，允许在定义类型注解时使用尚未在当前 Python 版本中引入的特性
import argparse         #argparse 模块用于解析命令行参数和选项
import sys              #sys 模块提供对解释器使用或维护的一些变量和函数的访问
import time             #time 模块提供时间相关的函数
from pathlib import Path #pathlib 模块提供面向对象的文件系统路径操作
from urllib.parse import urljoin, urlparse #urllib.parse 模块提供用于解析和构建 URL 的函数

import requests         #requests 模块用于发送 HTTP 请求
from bs4 import BeautifulSoup   #BeautifulSoup 模块用于解析 HTML 和 XML 文档


ROOT = Path(__file__).resolve().parents[2]  # 项目根目录
RAW_DIR = ROOT / "data" / "raw"          # 原始 HTML 存放目录
URLS_FILE = ROOT / "data" / "urls.txt"  # URL 列表文件

# 目标：Python 官方中文教程目录页
# 这个目录列出 Tutorial 的所有章节，每章是一个独立 HTML 页面
CATALOG_URL = "https://docs.python.org/zh-cn/3/tutorial/"

# Python 文档的章节链接在 toctree-wrapper 容器内
LINK_SELECTOR = "div.toctree-wrapper a"  

# 只保留同域名链接
SAME_DOMAIN_ONLY = True

# 只抓 tutorial 路径下的页面，避免误抓 library/reference 等其他部分
PATH_PREFIX =  "/zh-cn/3/tutorial/"
UA = "rag-hybrid-retrieval-qa/0.1 (learning project; contact: 2596468147@qq.com)"  #UA 字符串，用于标识请求来源

def session() -> requests.Session:
    """创建一个带 UA 的 requests 会话"""
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s

def discover() ->list[str]:
    """抓目录页 → 抽链接 → 去重 → 按 URL 字典序排序。"""
    if not CATALOG_URL or not LINK_SELECTOR:
       sys.exit("先把 CATALOG_URL 和 LINK_SELECTOR 填上") #这一步检查是否设置了目录页 URL 和链接选择器，如果没有设置则退出程序并提示用户填写。上面已经写了

    resp = session().get(CATALOG_URL,timeout=20)  # 使用自定义的 requests 会话发送 GET 请求获取目录页内容
    resp.raise_for_status()  # 如果请求失败，抛出异常
    soup = BeautifulSoup(resp.text, "html.parser")  # 使用 BeautifulSoup 解析

    base_domain = urlparse(CATALOG_URL).netloc  # 获取目录页的域名并取名为 base_domain，用于后续过滤同域名链接
    links: set[str] =set()  # 创建空集合，存放链接，自动去除重复 URL

    for a in soup.select(LINK_SELECTOR):
        href = a.get("href")
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        url = urljoin(CATALOG_URL, href).split("#")[0].rstrip("/")
        #三层过滤三层过滤：不是同域名 → 跳过 不在 tutorial 路径 → 跳过 目录页本身 → 跳过，不重复抓合格的 URL 加入集合 `links`。
        if SAME_DOMAIN_ONLY and urlparse(url).netloc != base_domain:
            continue
        if PATH_PREFIX and PATH_PREFIX not in urlparse(url).path:
            continue
        if url.rstrip("/") == CATALOG_URL.rstrip("/"):
            continue
        links.add(url) 

    urls = sorted(links)  #按字典序排序，后面文档编号稳定
    URLS_FILE.parent.mkdir(parents=True, exist_ok=True)  #自动创建 `data/` 目录
    URLS_FILE.write_text("\n".join(urls) + "\n", encoding="utf-8") #一行一个 URL，写入 `urls.txt`

    print(f"探测到 {len(urls)} 个章节链接")
    print("前 3 条样例：")
    for u in urls[:3]:
        print("  " + u)

      # Python 教程只有十几章，不是几百篇，所以不再做 800 篇警告
    if len(urls) < 5:
        print(
            "\n[警告] 少于 5 篇。检查 LINK_SELECTOR 是否匹配到了正确的容器，"
            "或者 PATH_PREFIX 是否把链接全过滤掉了。"
        )
    return urls

def read_urls() -> list[str]:           #读取链接清单,读 `urls.txt`，逐行去掉空白，过滤空行，返回 URL 列表。文件不存在就提示先跑 discover
    if not URLS_FILE.exists():
        sys.exit("data/urls.txt 不存在，先跑 --discover")
    return [ln.strip() for ln in URLS_FILE.read_text(encoding="utf-8").splitlines()if ln.strip()]

def fetch_all(limit: int | None = None) -> None:  #下载网页
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
        flag = "  <-- 偏小，疑似反爬页/JS空壳" if size < 10000 else ""
        print(f"[OK] {doc_id} {size // 1024} KB{flag}")
        got += 1
        time.sleep(1.0)  # 对文档站保持 1 秒间隔，体现合规

    print(f"\n完成：新抓 {got} 篇，跳过 {skipped} 篇，失败 {len(failed)} 篇")
    if failed:
        print("失败清单（前 5 条）：")
        for item in failed[:5]:
            print("  ", item)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true", help="只探测目录页并写 urls.txt")
    ap.add_argument("--limit", type=int, default=None, help="只抓前 N 篇，验收用")
    ap.add_argument(
        "--discover-then-fetch",
        action="store_true",
        help="先探测再立即抓取（新手便捷模式）",
    )
    args = ap.parse_args()

    if args.discover:
        discover()
    elif args.discover_then_fetch:
        discover()
        fetch_all(args.limit)
    else:
        fetch_all(args.limit)


if __name__ == "__main__":
    main()

