"""语料抓取：只负责把原文原样存到 data/raw/，不做任何清洗。

清洗是 parse.py 的事。两段分开跑，清洗规则会改很多遍，
分离之后改规则只需重跑解析，不用重爬。

支持的源见下面 SOURCES 表，第一种是纯目录页（抽链接），第二种是 GitHub 仓库
（走 API 列文件树），第三种是把文档页映射到 Sphinx 的 rst 源文件。加一个源
就是加一条配置，解析逻辑不用动。

落盘规则由 layout.py 决定（格式 / 域名 / 源路径），和 parse.py 共用同一个实现，
不会出现「爬下来的路径和解析时推的路径对不上」。

断点续爬靠 data/manifest.jsonl：它记着每个源地址抓到了哪个路径，
重跑时先查账本，已经落地的直接跳过。**判断依据是 URL，不是列表位置** ——
这条是踩过两次坑之后改的：以前 doc_id 用 enumerate 的位置，
换个源重新 discover，新链接插到前面，旧的编号全部错位，而且一声不响。

用法（PowerShell / bash，项目根目录下）：
    python -m src.ingest.crawl --sources                  # 列出所有可抓的源
    python -m src.ingest.crawl --source python-library --discover   # 只探测，写完清单
    python -m src.ingest.crawl --source python-library --limit 3    # 抓 3 篇（验收用）
    python -m src.ingest.crawl --source python-library              # 全量抓，断点续爬
    python -m src.ingest.crawl --source python-library --from-urls   # 按清单重建（不探测）
    python -m src.ingest.crawl --source django-rst --limit 2        # 抓 Django 的 rst 源
"""

from __future__ import annotations  #解释这一行：启用未来风格的类型注解，允许在定义类型注解时使用尚未在当前 Python 版本中引入的特性
import argparse         #argparse 模块用于解析命令行参数和选项
import json             #json 模块用于读写 manifest
import sys              #sys 模块提供对解释器使用或维护的一些变量和函数的访问
import time             #time 模块提供时间相关的函数
from datetime import datetime, timezone
from pathlib import Path #pathlib 模块提供面向对象的文件系统路径操作
from urllib.parse import urljoin, urlparse, urlunparse #urllib.parse 模块提供用于解析和构建 URL 的函数

import requests         #requests 模块用于发送 HTTP 请求
from bs4 import BeautifulSoup   #BeautifulSoup 模块用于解析 HTML 和 XML 文档

from .formats import decode, detect_format
from .layout import rel_path_of

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录
RAW_DIR = ROOT / "data" / "raw"          # 原始文件存放目录
URLS_DIR = ROOT / "data" / "urls"        # 每个源一份清单，互不覆盖
URLS_FILE = ROOT / "data" / "urls.txt"   # 所有源合并后的清单，供 parse 校验
MANIFEST_FILE = ROOT / "data" / "manifest.jsonl"  # 抓取账本：源地址 → 落地路径

UA = "rag-hybrid-retrieval-qa/0.1 (learning project; contact: 2596468147@qq.com)"  #UA 字符串，用于标识请求来源
SLEEP = 1.0  # 对文档站保持 1 秒间隔，体现合规

'''
后注（历史，别删）：
第一次语料搜集选的是：https://docs.python.org/zh-cn/3/tutorial/，即原版python教程，共16篇
第二次选取的是：https://docs.python.org/zh-cn/3/library/，python标准库的内容，共250篇
第三次开始用 SOURCES 表，一个源一条配置，不再靠改全局常量切换。
'''

# ---------------- 源地址改写 ----------------

def django_sources_url(page_url: str) -> str:
    """Django 文档页 → 它的 rst 源文件。

    Sphinx 把每个页面的源码放在 /_sources/ 下，扩展名统一是 .txt（内容其实是 rst）：
        https://docs.djangoproject.com/zh-hans/5.2/topics/db/models/
        https://docs.djangoproject.com/zh-hans/5.2/_sources/topics/db/models.txt

    直接抓渲染后的 HTML 也能用，但源码里没有 headerlink、没有 ¶、没有侧栏，
    标题层级是显式的（下划线），解析质量明显更高，体积也只有一半左右。
    """
    p = urlparse(page_url)
    path = p.path.rstrip("/").replace("/_sources/", "/")   # 防止重复插入
    path = path.replace("/5.2/", "/5.2/_sources/", 1)
    return urlunparse((p.scheme, p.netloc, path + ".txt", "", "", ""))

# ---------------- 源配置 ----------------

# 每条源的字段：
#   kind        catalog = 抓目录页抽链接；github = 走 GitHub API 列文件树
#   catalog     目录页地址（kind=catalog）
#   selector    目录页里章节链接所在容器（Sphinx 是 div.toctree-wrapper）
#   path_prefix 只保留路径含该前缀的链接，防止误抓同站其他栏目
#   fmt         落地格式：html / rst / md
#   rewrite     （可选）页面地址 → 实际请求地址的函数
#   repo/branch/path_filter   GitHub 源的仓库、分支、只抓哪些子目录
SOURCES = {
    "python-library": {
        "kind": "catalog",
        "catalog": "https://docs.python.org/zh-cn/3/library/",
        "selector": "div.toctree-wrapper a",
        "path_prefix": "/zh-cn/3/library/",
        "fmt": "html",
    },
    "python-tutorial": {
        "kind": "catalog",
        "catalog": "https://docs.python.org/zh-cn/3/tutorial/",
        "selector": "div.toctree-wrapper a",
        "path_prefix": "/zh-cn/3/tutorial/",
        "fmt": "html",
    },
    "python-reference": {
        "kind": "catalog",
        "catalog": "https://docs.python.org/zh-cn/3/reference/",
        "selector": "div.toctree-wrapper a",
        "path_prefix": "/zh-cn/3/reference/",
        "fmt": "html",
    },
    "python-c-api": {
        "kind": "catalog",
        "catalog": "https://docs.python.org/zh-cn/3/c-api/",
        "selector": "div.toctree-wrapper a",
        "path_prefix": "/zh-cn/3/c-api/",
        "fmt": "html",
    },
    "django-rst": {
        "kind": "catalog",
        "catalog": "https://docs.djangoproject.com/zh-hans/5.2/ref/",
        "selector": "div.toctree-wrapper a",
        "path_prefix": "/zh-hans/5.2/",
        "fmt": "rst",
        "rewrite": django_sources_url,
        # Django 的目录页只列顶层（ref 28 条），真正的页面在子目录里。
        # 想抓全就换 catalog 到 topics/ 或 howto/，各跑一次，清单会自动合并。
    },
    "django-html": {
        "kind": "catalog",
        "catalog": "https://docs.djangoproject.com/zh-hans/5.2/ref/",
        "selector": "div.toctree-wrapper a",
        "path_prefix": "/zh-hans/5.2/",
        "fmt": "html",
        # Django 的 canonical 指向**最新版本**（5.2 的页面自称 6.1），
        # 拿它定路径会让所有文件落到 6.1 目录、doc_id 也跟着错版本。
        # 这类站的路径只能按请求地址算。
        "use_canonical": False,
    },
    "python100days-md": {
        "kind": "github",
        "repo": "jackfrued/Python-100-Days",
        "branch": "master",
        # 只取按天组织的教程正文。根目录 README 是课程索引（一堆链接），
        # 首页那种东西切出来全是碎片，不值当进语料。
        "path_filter": ["Day", "番外篇/"],
        "fmt": "md",
    },
    "vue-docs-md": {
        "kind": "github",
        "repo": "vuejs-translations/docs-zh-cn",
        "branch": "main",
        # 只取指南和 API 两部分。about/ 是行为准则和社区守则，
        # releases/ 是更新日志，api/ 之外的 tutorials 是重复内容 —— 都是噪声
        "path_filter": ["src/guide/", "src/api/"],
        "fmt": "md",
    },
}


def session() -> requests.Session:
    """创建一个带 UA 的 requests 会话"""
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


# ---------------- 目录页 / 仓库 → 待抓清单 ----------------

def discover_catalog(src: dict) -> list[dict]:
    """抓目录页 → 抽链接 → 去重 → 返回 [{page, fetch, fmt}]。

    page 是「人能点开的页面地址」，fetch 是「实际请求的地址」。
    两者在 html 源里相同；在 django-rst 源里 page 是文档页、fetch 是 _sources 里的 .txt。
    引用溯源要用 page，下载要用 fetch，混了就会拿源码地址当引用、点开是纯文本。
    """
    catalog = src["catalog"]
    resp = session().get(catalog, timeout=20)  # 使用自定义的 requests 会话发送 GET 请求获取目录页内容
    resp.raise_for_status()  # 如果请求失败，抛出异常
    soup = BeautifulSoup(resp.text, "html.parser")  # 使用 BeautifulSoup 解析

    base_domain = urlparse(catalog).netloc  # 获取目录页的域名并取名为 base_domain，用于后续过滤同域名链接
    prefix = src.get("path_prefix", "")
    rewrite = src.get("rewrite")
    fmt = src["fmt"]
    links: dict[str, dict] = {}   # page → item，用 dict 去重且保留插入顺序

    for a in soup.select(src["selector"]):
        href = a.get("href")
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        page = urljoin(catalog, href).split("#")[0].rstrip("/")
        #三层过滤：不是同域名 → 跳过；不在目标路径 → 跳过；目录页本身 → 跳过
        if urlparse(page).netloc != base_domain:
            continue
        if prefix and prefix not in urlparse(page).path:
            continue
        if page.rstrip("/") == catalog.rstrip("/"):
            continue
        fetch = rewrite(page) if rewrite else page
        links[page] = {"page": page, "fetch": fetch, "fmt": fmt}

    return list(links.values())


def discover_github(src: dict) -> list[dict]:
    """GitHub 仓库 → 仓库里的 .md 文件清单。

    一次递归 tree 请求就能拿到全仓文件列表（未认证 60 次/小时够用几百次），
    之后下载走 raw.githubusercontent.com，**不计入 API 限额**。

    为什么不 git clone：这类文档仓动辄几百 MB，大半是图片和 node_modules 时代的
    历史遗留。只取 .md，一个 300 MB 的仓能缩到 3 MB。
    """
    api = f"https://api.github.com/repos/{src['repo']}/git/trees/{src['branch']}?recursive=1"
    resp = session().get(api, timeout=30)
    if resp.status_code != 200:
        print(f"[警告] GitHub API 返回 {resp.status_code}，可能是限额用完了（未认证 60 次/小时）")
        return []
    data = resp.json()
    if data.get("truncated"):
        print("[警告] 文件树被截断（仓库太大），只拿到一部分")

    prefix = src.get("path_filter", "")
    # path_filter 允许写成列表：GitHub 仓里往往只有几个子目录值得抓
    # （Vue 的 about/ 是行为准则、releases/ 是更新日志，抓了就是噪声）
    prefixes = [prefix] if isinstance(prefix, str) else list(prefix or [])
    fmt = src["fmt"]
    owner_repo = src["repo"].replace("/", "__")
    out: list[dict] = []
    for node in data.get("tree", []):
        if node.get("type") != "blob":
            continue
        path = node["path"]
        if not path.endswith(".md"):
            continue
        if prefixes and not any(path.startswith(p) for p in prefixes):
            continue
        url = f"https://raw.githubusercontent.com/{src['repo']}/{src['branch']}/{path}"
        # 落地路径用 github.com/{owner__repo}/... 这种形态，而不是 raw.githubusercontent.com。
        # 理由：raw 是 CDN 域名，用户点开引用要的是仓库页面；而且 owner__repo 合成一段
        # 之后，同一个仓库的所有文档在 raw 下自然聚成一棵树。
        out.append({
            "page": f"https://github.com/{src['repo']}/blob/{src['branch']}/{path}",
            "fetch": url,
            "fmt": fmt,
            "layout_url": f"https://github.com/{owner_repo}/{path}",
        })
    return out


def discover(src: dict) -> list[dict]:
    """按源的 kind 分发到对应的发现逻辑。"""
    if src["kind"] == "github":
        return discover_github(src)
    return discover_catalog(src)


# ---------------- 清单与账本 ----------------

def write_urls(name: str, items: list[dict], replace: bool = False) -> list[str]:
    """写本源的清单，并重建全体并集 data/urls.txt。

    每个源一份清单（data/urls/{源名}.txt）而不是共用一份，是因为以前共用时
    discover 一跑就把上一批的清单整份覆盖掉 —— 换源等于把旧账撕了。
    分文件之后，想抓哪个源就跑哪个，互不干扰。

    **同一个源换栏目跑的时候默认合并**：Django 的 topics / ref / howto 各是一棵
    独立的目录树，只能一个栏目一个栏目抓。如果每次 discover 都覆盖，
    抓完 topics 再抓 ref，topics 那批的清单就没了，断点续爬也就废了。
    要真正重置（比如换了版本号）传 replace=True。
    """
    URLS_DIR.mkdir(parents=True, exist_ok=True)
    path = URLS_DIR / f"{name}.txt"

    existing: set[str] = set()
    if path.exists() and not replace:
        existing = {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}

    fresh = {it["page"] for it in items}
    added = fresh - existing
    merged = sorted(existing | fresh)
    path.write_text("\n".join(merged) + "\n", encoding="utf-8")

    # 全体并集，供 parse.py 判断「这篇在不在抓取清单里」
    all_urls: set[str] = set()
    for f in URLS_DIR.glob("*.txt"):
        all_urls.update(ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
    URLS_FILE.write_text("\n".join(sorted(all_urls)) + "\n", encoding="utf-8")

    print(f"  清单：本次发现 {len(fresh)} 条，新增 {len(added)} 条，累计 {len(merged)} 条")
    return merged


def load_manifest() -> dict[str, dict]:
    """读抓取账本，返回 {page_url: 记录}。"""
    if not MANIFEST_FILE.exists():
        return {}
    out: dict[str, dict] = {}
    for ln in MANIFEST_FILE.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if rec.get("url"):
            out[rec["url"]] = rec
    return out


def save_manifest(records: dict[str, dict]) -> None:
    """整份重写账本。按 URL 去重，按落地路径排序，方便人肉 diff。"""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(records.values(), key=lambda r: r.get("path", ""))
    with MANIFEST_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- 下载 ----------------

def canonical_of(html: bytes) -> str:
    """从下载到的 HTML 里取 canonical。取不到返回空串。

    HTML 的落地路径要用它，而不是用请求地址 —— 请求地址带语言前缀，
    canonical 不带，两套路径会让同一页存成两份（这个坑在迁移 raw 时已经吃过一次）。
    """
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("link", attrs={"rel": ["canonical"]})
    if tag and tag.get("href"):
        return tag["href"].strip()
    meta = soup.find("meta", attrs={"property": "og:url"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    return ""


def items_from_urls(name: str) -> list[dict]:
    """不联网重建待抓条目：直接读本源已落盘的清单。

    为什么需要这件事：
        catalog 源的 discover 只能拿到「当前目录页上挂着的链接」。Django 是分栏目
        抓的（topics / ref / howto 各跑一次），每次 discover 只返回当前栏目的几十条，
        清单里另外一百多篇根本不在返回结果里。结果就是：raw 被删掉之后，只靠
        discover 重跑会少一百多篇，而且一句报错都没有 —— 你在 parse 的报告里
        看到篇数变少，会先怀疑代码，其实只是没抓。
        build.py 要的「一条命令从零重建」，前提是按清单重下，而不是重新猜清单。

    github 源例外：清单里存的是仓库页面地址，真正的下载地址（raw CDN 直链）
    要过 API 拿，没有 API 就拼不出来，所以那类源仍然走 discover。它们的条目数
    由仓库文件树决定，重新探测也不会少。
    """
    fmt = SOURCES[name]["fmt"]
    rewrite = SOURCES[name].get("rewrite")
    path = URLS_DIR / f"{name}.txt"
    if not path.exists():
        return []

    items: list[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        page = ln.strip()
        if not page:
            continue
        items.append({"page": page, "fetch": rewrite(page) if rewrite else page, "fmt": fmt})
    return items


def fetch_all(name: str, limit: int | None = None, force: bool = False,
              replace: bool = False, from_urls: bool = False) -> dict:
    """下载一个源的全部条目，已落地的自动跳过（断点续爬）。

    from_urls=True 时不探测目录页，按 data/urls/{源}.txt 重建 —— 复原语料用这个，
    扩充语料才需要重新探测。
    """
    src = SOURCES[name]
    rebuilt = from_urls and src["kind"] == "catalog"
    if rebuilt:
        items = items_from_urls(name)
        if items:
            print(f"  按清单重建 {len(items)} 条（不探测目录页）")
        else:
            print(f"  [提示] {name} 还没有清单，改为探测目录页")
            rebuilt = False
    if not rebuilt:
        items = discover(src)
        if not items:
            sys.exit(f"源 {name} 没发现任何链接。检查 selector / path_prefix / 仓库名。")
        write_urls(name, items, replace)
    if limit:
        items = items[:limit]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    sess = session()
    got, skipped, failed, warns = 0, 0, [], 0
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    for i, it in enumerate(items, 1):
        page, fetch, fmt = it["page"], it["fetch"], it["fmt"]

        # 账本里有、且文件还在 → 真的抓过了，跳过
        old = manifest.get(page)
        if old and not force and (RAW_DIR / old.get("path", "")).exists():
            skipped += 1
            continue

        try:
            resp = sess.get(fetch, timeout=30)
            if resp.status_code == 404 and fmt == "rst" and fetch.endswith(".txt"):
                # Sphinx 的目录型页面（有子页面、自己没有正文的那种），
                # 源文件不在 xxx.txt 而在 xxx/index.txt。这不算失败，换个地址就行。
                alt = fetch[:-4] + "/index.txt"
                resp = sess.get(alt, timeout=30)
                if resp.status_code == 200:
                    fetch = alt
            resp.raise_for_status()
        except Exception as e:  # 单个失败不该中断整批
            failed.append((page, repr(e)))
            print(f"[FAIL] {page} -> {e!r}")
            continue

        data = resp.content
        if not data.strip():
            failed.append((page, "空响应"))
            print(f"[FAIL] {page} -> 空响应")
            continue

        # 格式以内容为准，不以配置为准：配置写 rst 但服务器返回了一张 HTML 报错页，
        # 按配置存下去会在 parse 阶段变成一坨导航文本混进语料。
        # 但 text 是嗅探的兜底结论，不代表矛盾 —— 只拒绝「明确是另一种格式」的情况。
        actual = detect_format(data)
        if src["kind"] != "github" and actual != fmt and actual != "text":
            failed.append((page, f"内容实际是 {actual}，与配置的 {fmt} 不符"))
            print(f"[FAIL] {page} -> 内容实际是 {actual}，不是 {fmt}")
            continue
        if actual == "text" and fmt != "text":
            # 嗅探没把握时按源配置走，但打个标记方便事后查
            print(f"[注意] {page} 嗅探结果是 text，按配置的 {fmt} 存")

        # HTML 用页面自己声明的 canonical 定路径；其余格式用实际请求地址
        # layout_url 是给「下载地址和文档地址不是一回事」的源准备的：
        # GitHub 的 raw CDN 地址不适合当路径（用户点开要的是仓库页面）
        # use_canonical=False 的源（Django）一律按请求地址，因为它的 canonical
        # 指向最新版本，不等于你正在抓的这一版
        use_canon = src.get("use_canonical", True)
        authority = it.get("layout_url") or (
            canonical_of(data) if (actual == "html" and use_canon) else fetch
        )
        if actual == "html" and not authority:
            warns += 1
            authority = fetch      # 没有 canonical 就用请求地址兜底，并在最后报出来
        # Sphinx 的 /_sources/ 是部署细节，不属于文档路径。留在路径里会让 raw
        # 看起来像在备份一个目录结构，而不是在爬文档 —— 引用溯源时也读着别扭。
        # 落盘格式一律按源配置，不按嗅探结果：嗅探只是发现异常的手段，
        # 一次误判（144 字节的短 md 被认成 rst）就足以把文件塞进错误的格式目录。
        save_fmt = fmt
        rel = rel_path_of(authority.replace("/_sources/", "/"), save_fmt)

        out = RAW_DIR / rel
        if out.exists() and not force:
            # 文件已在（可能上次写盘后账本没来得及更新），补一条账就走
            manifest[page] = {"path": rel, "url": page, "fetch_url": fetch,
                              "format": save_fmt, "source": name, "at": now}
            skipped += 1
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        # 存字节，不存 str。编码问题留给 parse.py 处理，raw 保持原样
        out.write_bytes(data)
        manifest[page] = {"path": rel, "url": page, "fetch_url": fetch,
                          "format": save_fmt, "source": name, "at": now}

        size = len(data)
        flag = "  <-- 偏小，疑似反爬页/JS空壳" if size < 2000 else ""
        print(f"[OK] {size // 1024:>4} KB  {rel}{flag}")
        got += 1
        time.sleep(SLEEP)  # 对文档站保持 1 秒间隔，体现合规

        if i % 25 == 0:
            save_manifest(manifest)   # 定期落账，中途被打断不至于全丢

    save_manifest(manifest)
    print(f"\n完成：新抓 {got} 篇，跳过 {skipped} 篇，失败 {len(failed)} 篇")
    if warns:
        print(f"[提示] {warns} 篇没有 canonical，路径按请求地址兜底")
    if failed:
        print("失败清单（前 5 条）：")
        for page, why in failed[:5]:
            print(f"  {page}\n    {why}")

    # 返回计数，给 scripts/build.py 汇总用。只打印不返回的话，
    # 上层想知道「这次到底抓成功几篇」只能去截自己的 stdout，太脆。
    return {"source": name, "planned": len(items), "got": got,
            "skipped": skipped, "failed": failed, "warns": warns}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", action="store_true", help="列出所有可抓的源")
    ap.add_argument("--source", default=None, help="选一个源，见 --sources")
    ap.add_argument("--catalog", default=None,
                    help="临时覆盖该源的目录页地址（只对 catalog 类源有效）")
    ap.add_argument("--discover", action="store_true", help="只探测目录页并写清单，不下载")
    ap.add_argument("--limit", type=int, default=None, help="只抓前 N 篇，验收用")
    ap.add_argument("--force", action="store_true", help="已存在的也重抓")
    ap.add_argument("--replace", action="store_true",
                    help="重置本源清单（默认与已有清单合并，换栏目抓的时候别加这个）")
    ap.add_argument("--from-urls", action="store_true",
                    help="不探测目录页，按 data/urls/{源}.txt 重建（复原语料用）")
    args = ap.parse_args()

    if args.sources or not args.source:
        print("可抓的源：")
        for name, s in SOURCES.items():
            where = s.get("catalog") or s.get("repo")
            print(f"  {name:<18} {s['fmt']:<5} {where}")
        print("\n用法：python -m src.ingest.crawl --source <名字> [--discover|--limit N]")
        return

    if args.source not in SOURCES:
        sys.exit(f"没有这个源：{args.source}。先跑 --sources 看看有哪些。")

    if args.catalog:
        # 临时换目录页：比如 Django 想抓 topics 那一块，不必改配置表
        SOURCES[args.source] = {**SOURCES[args.source], "catalog": args.catalog,
                                "path_prefix": urlparse(args.catalog).path}

    if args.discover:
        items = discover(SOURCES[args.source])
        urls = write_urls(args.source, items, args.replace)
        print(f"探测到 {len(items)} 个链接，清单累计 {len(urls)} 条（data/urls/{args.source}.txt）")
        for u in urls[:3]:
            print("  " + u)
        if len(items) < 5:
            print("\n[警告] 少于 5 篇。检查 selector 是否匹配到正确的容器，"
                  "或者 path_prefix 是否把链接全过滤掉了。")
        return

    fetch_all(args.source, args.limit, args.force, args.replace, args.from_urls)


if __name__ == "__main__":
    main()
