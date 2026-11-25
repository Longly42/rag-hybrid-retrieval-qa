"""语料清洗与切分：data/raw/**/* → data/corpus.jsonl，一行一个 chunk。

为什么不一一行一篇文档：
    检索的最小单位是 chunk，不是文档。一行一篇的话，第 3 步的重排、
    第 6 步的引用溯源全部做不了，等于推倒重来。

为什么按 <section> 递归（HTML）：
    Sphinx 用 <section id=...> 把「标题 + 该标题下的内容」包成一个整体，
    嵌套 section 就是天然的标题树。递归进去就拿到了 title_path。

支持哪些格式：
    html  —— BeautifulSoup + 站点 profile（Sphinx 那套 section/dl/dt 逻辑）
    rst   —— src/ingest/formats.py，Sphinx 源文件，Django 的 _sources 就是
    md    —— src/ingest/formats.py，GitHub 上的中文文档仓
    text  —— 兜底，按空行分段
    格式靠**内容嗅探**判定，不看扩展名。实测反例：Django 的源文件叫 .txt 内容
    是 rst；CS-Notes 的 README.md 里混着 HTML 标签。

URL 的来源：
    HTML 用页面里的 <link rel="canonical">；但展示地址要用带语言前缀的，
    因为 canonical 会丢掉 /zh-cn。
    rst / md 没有 canonical，从 manifest.jsonl 取（crawl 落下来的 path→url 映射），
    没有 manifest 时按 raw 的目录结构反推。

扫描范围：
    递归扫 data/raw/ 下所有文件，不预设站点、不预设栏目。raw 的目录约定是
    「格式 / 域名 / 源路径」三层，解析时这三段分别变成 format / site / source_path。

同一篇文档被抓了两份（html 和 rst）怎么办：
    按「域名 + 路径」归一化后只留一份，优先 rst > md > html > text。
    源码比渲染后的页面干净，这是唯一需要人为排序的地方。

用法：
    python -m src.ingest.parse                    # 扫全部，不筛栏目
    python -m src.ingest.parse --limit 10         # 只切前 10 篇，验收用
    python -m src.ingest.parse --section library  # 只要 library 栏目
    python -m src.ingest.parse --format rst       # 只处理某种格式
    python -m src.ingest.parse --prefix /library/ # 按路径前缀筛（旧参数，保留）
    python -m src.ingest.parse --keep-toc         # 不丢弃目录页
    python -m src.ingest.parse --report out.json  # 把统计写成 JSON，数字要留档
"""

from __future__ import annotations #导入未来定义的类,作用：允许在类定义之前使用类名
import argparse   #导入命令行解析模块
import hashlib    #导入哈希模块用于生成文档的哈希值
import json       #导入json模块用于处理json文件
import re         #导入正则表达式模块
import sys        #导入系统模块
from datetime import datetime #导入时间模块用于给统计报告打时间戳
from pathlib import Path #导入路径模块
from urllib.parse import urlparse #导入urlparse模块用于解析url

from bs4 import BeautifulSoup #导入BeautifulSoup模块用于解析html文件

from .formats import Block, decode, detect_format, parse_markup
from .layout import url_of_rel

#解释下列变量：ROOT：项目根目录，RAW_DIR：原始数据目录，URLS_FILE：URL文件路径，OUT_FILE：输出文件路径
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
URLS_FILE = ROOT / "data" / "urls.txt"
OUT_FILE = ROOT / "data" / "corpus.jsonl"
MANIFEST_FILE = ROOT / "data" / "manifest.jsonl"

# ---------- 切分参数 ----------
CHUNK_MAX = 600        # 单 chunk 字符上限
CHUNK_OVERLAP = 90     # 相邻 chunk 字符重叠数
MIN_PAGE_CHARS = 200   # 整页正文短于此 → 丢弃（原 300 略微放宽）
MIN_CHUNK_CHARS = 60   # 单块短于此 → 丢弃
CODE_MAX = 1500        # 代码块超过此长度 → 按行折开

# ---------- 目录页判定参数 ----------
# 目录页 = 正文很短、没有任何 h2/h3、内容基本都是链接标题。它答不了任何问题，
# 却会把 "itertools"、"hashlib" 这类词送进索引，抢走真正答案的位置。
TOC_MAX_CHARS = 800    # 正文超过这个长度就不当目录页看
TOC_LINK_RATIO = 0.5   # 链接文字占正文的比例超过它，判为目录页

# ---------- 格式优先级 ----------
# 同一篇文档有多份时，留哪份。源码 > 渲染后的 HTML，因为源码里没有导航、
# 没有 headerlink、没有 § 锚点，标题层级是显式的。
FORMAT_PRIORITY = {"rst": 3, "md": 2, "html": 1, "text": 0}

# ---------- 正则 ----------
#解释下列变量：HEADING_RE：标题正则表达式，SENT_SPLIT_RE：句子分割正则表达式，VERSION_RE：版本号正则表达式
HEADING_RE = re.compile(r"^h[1-6]$")               
SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])|(?<=[.!?])\s+|\n+")
VERSION_RE = re.compile(r"Python\s+([\d.]+)")
# 路径里的语言代码段（zh-cn / zh-hans / en-us / ja）与版号段（3 / 3.15 / v2），推断栏目名时跳过
# 语言段用白名单，不能用「两三个小写字母」这种形状规则 ——
# Django 的 ref 目录会被形状规则当成语言码跳掉，section 就从 ref 变成 models。
# 语言码是可枚举的，栏目名不是。
LANG_SEG_RE = re.compile(
    r"^(zh|en|ja|ko|fr|de|es|pt|ru|it|nl|pl|tr|ar|hi|vi|th|id)([-_][a-zA-Z]{2,8})?$"
)
VER_SEG_RE = re.compile(r"^v?\d+(\.\d+)*$")
# GitHub raw 的路径是 /{owner}/{repo}/{branch}/...，前两段合起来才是「项目」，
# 只取第一段会得到用户名，作为 section 没有意义
GITHUB_RAW_HOST = "raw.githubusercontent.com"
# 文档路径的扩展名，算 doc_key 时要剥掉
DOC_EXT_RE = re.compile(r"\.(html?|txt|md|rst|text)$", re.I)

# 站点配置，因为不同站点的 HTML 结构不同
# 新增一个站：在这里加一条 {域名: {body: 正文选择器, drop: 要删的噪声选择器}} 即可，
# 不改任何逻辑代码。
SITE_PROFILES = {
    "docs.python.org": {
        "body": "div.body",
        "drop": ["div.sphinxsidebar", "div.related", "div.footer", "div.clearer"],
        # canonical 会丢掉语言前缀，这里记下正确的语言段，拼引用地址时补回去。
        # 不补的话，编号错位的那些页会拿到英文地址，引用溯源会跳到英文页。
        "lang_prefix": "/zh-cn",
    },
    "docs.djangoproject.com": {
        # Django 用自建模板，没有 div.body，正文容器是 article
        "body": "article",
        "drop": ["div.sidebar", "nav", "div#doc-versions", "div#custom-doc-nav"],
        "lang_prefix": "/zh-hans",
    },
}

# 未登记的站点用这个兜底。选择器写成逗号列表，select_one 会按顺序取第一个命中的。
# 覆盖 Sphinx / MkDocs(main|article) / Docusaurus(article) / VitePress(.vp-doc) 这几类常见博客与文档站
DEFAULT_PROFILE = {
    "body": "div.body, article, main, div.vp-doc, div.md-content, div#content, div.content, body",
    "drop": ["div.sphinxsidebar", "div.related", "div.footer", "div.clearer",
             "div.sidebar", "aside", "footer", "header"],
}

DROP_ALWAYS = ["script", "style", "nav", "a.headerlink"] # DROP_ALWAYS：始终丢弃的标签列表
# 视为「一个内容块」的标签
BLOCK_TAGS = {"p", "ul", "ol", "dl", "table", "blockquote", "pre", "div"}

# inline 标签：抽文字前要先 unwrap 掉
INLINE_TAGS = ["a", "code", "tt", "em", "strong", "span", "b", "i",
               "sup", "sub", "abbr", "kbd", "var", "dfn", "samp", "cite"]



# ------工具函数--------
def read_urls() -> list[str]:
    """读取 URLS_FILE，返回 URL 列表。"""
    if not URLS_FILE.exists():
        return []
    return [ln.strip() for ln in URLS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]

def load_manifest() -> dict[str, str]:
    """读 data/manifest.jsonl，返回 {raw 相对路径: 源 URL}。

    manifest 是 crawl.py 落下来的抓取账本。没有它也能跑（URL 从目录结构反推），
    但有它才能保证引用地址和真实请求地址一字不差 —— 尤其是 Django 那种
    目录式 URL，反推出来会少一个结尾斜杠。
    """
    if not MANIFEST_FILE.exists():
        return {}
    out: dict[str, str] = {}
    for ln in MANIFEST_FILE.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue           # 半行写入的坏记录，跳过就好，不该让整次解析失败
        p, u = rec.get("path"), rec.get("url")
        if p and u:
            out[p.replace("\\", "/")] = u
    return out

def pick_profile(url: str) -> tuple[dict, bool]:
    """根据 URL 选择站点配置。

    返回 (配置, 是否是已登记的站点)。第二个值用来在最后报告「有哪些域名没登记」，
    提醒你给它补一条 profile，否则它只能靠 DEFAULT_PROFILE 的通用选择器硬猜。
    """
    host = urlparse(url).netloc
    prof = SITE_PROFILES.get(host)
    if prof:
        return prof, True
    return DEFAULT_PROFILE, False

def basename(url: str) -> str:
    """从 URL 中提取文件名。"""
    return urlparse(url).path.rsplit("/", 1)[-1]

def section_of(url: str) -> str:
    """从 URL 路径推断栏目名，作为 chunk 的 section 字段。

    /zh-cn/3/library/argparse.html → library
    /3/library/argparse.html       → library
    /tutorial/intro/               → tutorial
    /index.html                    → ""（根页面，没有栏目）

    先去掉文件名，再去掉语言段和版号段，剩下第一段就是栏目。
    这样以后加 Django 的 /zh-hans/5.2/topics/ 会自动得到 topics，
    加 FastAPI 的 /tutorial/ 得到 tutorial，不需要改代码。
    """
    parts = [p for p in urlparse(url).path.split("/") if p]
    if parts and "." in parts[-1]:
        parts = parts[:-1]
    # GitHub 单独处理：section 取 owner__repo，否则会变成用户名（一个用户下十几个仓，
    # 全被归到同一个栏目，分域统计就没意义了）。
    # github.com/{owner}/{repo}/blob/{branch}/... 要前四段才认得出是仓库，
    # 只按前两段合并会把 github.com/{owner__repo}/README 这种误伤。
    host = urlparse(url).netloc
    if host == "raw.githubusercontent.com" and len(parts) >= 2:
        return f"{parts[0]}__{parts[1]}"
    if host == "github.com" and len(parts) >= 4 and parts[2] in ("blob", "tree", "raw"):
        return f"{parts[0]}__{parts[1]}"
    while parts and (LANG_SEG_RE.match(parts[0]) or VER_SEG_RE.match(parts[0])):
        parts = parts[1:]
    return parts[0] if parts else ""

def doc_key(url: str) -> str:
    """文档身份：域名 + 路径，去掉扩展名和尾斜杠。

    两个用途：
      1. 当 doc_id —— 它锚在 URL 上，不再依赖「文件在清单里排第几」，
         所以重爬多少次都不会错位（编号错位那个坑到此为止）。
      2. 判断「html 和 rst 是不是同一篇」—— 渲染地址和源文件地址去掉扩展名后
         是同一条路径，靠它就能识别出双份。
    """
    p = urlparse(url)
    path = DOC_EXT_RE.sub("", p.path).rstrip("/")
    return f"{p.netloc}{path}"

def site_of(url: str) -> str:
    """域名。分域建索引、分域评测时按它筛。"""
    return urlparse(url).netloc

def localized_url(url: str) -> str:
    """把 canonical 丢掉的语言前缀补回来，得到能直接打开的中文地址。

    canonical 是 https://docs.python.org/3/library/bisect.html，
    中文版实际在 https://docs.python.org/zh-cn/3/library/bisect.html。
    只有在站点 profile 里配了 lang_prefix 时才补，别的站原样返回 ——
    乱补会把地址改成 404。
    """
    prof, _ = pick_profile(url)
    pre = prof.get("lang_prefix")
    if not pre:
        return url
    p = urlparse(url)
    if p.path.startswith(pre):
        return url
    return f"{p.scheme}://{p.netloc}{pre}{p.path}"

def canonical_url(soup: BeautifulSoup) -> str | None:
    """从 HTML 中提取 canonical URL。"""
    tag = soup.find("link", attrs={"rel": ["canonical"]})
    if tag and tag.get("href"):
        return tag["href"].strip()
    meta = soup.find("meta", attrs={"property": "og:url"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    return None

def heading_text(h) -> str:
    """提取标题文本。"""
    if h is None:
        return ""
    return re.sub(r"\s+", " ", h.get_text(" ", strip=True).replace("¶", "")).strip()


def page_meta(soup: BeautifulSoup) -> tuple[str, str]:
    """提取页面标题和版本号。"""
    raw = (soup.title.string or "").strip() if soup.title else ""
    version = ""
    m = VERSION_RE.search(raw)
    if m:
        version = m.group(1)
    title = re.split(r"\s+[—–]\s+", raw)[0].strip()
    return title, version

def clean(soup: BeautifulSoup, profile: dict) -> None:
    """清洗 HTML。"""
    for sel in list(profile.get("drop", [])) + DROP_ALWAYS:
        for el in soup.select(sel):
            el.decompose()

def looks_like_toc(body) -> bool:
    """判断这一页是不是「只有链接」的目录页。

    三个条件同时成立才算：
      1. 正文短于 TOC_MAX_CHARS
      2. 没有 h2/h3/h4（h1 是页面标题，不算章节）
      3. 链接文字占正文比例超过 TOC_LINK_RATIO

    为什么不能只看字数：正文 200 字的正常小页面是有的（比如某个只有两段话的
    模块页）；为什么不能只看有没有 h2：有些正常页面也不分节。
    三个条件叠起来才不会误杀。
    """
    text = body.get_text(" ", strip=True)
    if len(text) >= TOC_MAX_CHARS:
        return False
    if body.find(["h2", "h3", "h4"]) is not None:
        return False
    link_chars = sum(len(a.get_text(strip=True)) for a in body.find_all("a"))
    return len(text) > 0 and link_chars >= len(text) * TOC_LINK_RATIO

# -----------抽块-------------
def pre_text(el) -> str:
    """ 用于 <pre> 标签的文本提取 """
    for go in el.select("span.go"):
        go.string = go.get_text().replace("\\n", "\n")
    lines = [ln.rstrip() for ln in el.get_text().splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)

def block_text(el) -> str:
    """ 用于非 <pre> 标签的文本提取 """
    for t in el.find_all(INLINE_TAGS):
        t.unwrap()
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()

def is_noise(text: str) -> bool:
    """ 判断文本是否为噪音 """
    return len(text) < 80 and text.startswith(("源代码:", "源码："))

def extend_path(path: tuple[str, ...], title: str) -> tuple[str, ...]:
    """ 扩展路径 """
    if not title:
        return path
    if path and path[-1] == title:
        return path
    return path + (title,)

def iter_blocks(node, path: tuple[str, ...], out: list[tuple[tuple[str, ...], str, str]],
                state: dict | None = None) -> None:
    """ 遍历节点，提取内容块

    state 用来在同一个 <dl> 里把 dt 的标题传给紧随其后的 dd。
    不传的话，dd 拿到的还是 dl 外层的 path，函数签名和它的说明就会被拆成
    两个不同的组、切成两个 chunk —— 说明那段还会失去归属，标题只剩模块名。
    """
    if state is None:
        state = {}
    for child in node.children:     #遍历，第一层是node，第二层是node的子节点，第三层是子节点的子节点
        name = getattr(child, "name", None)
        if name is None:
            continue
        if name in ("script", "style"):
            continue
        if HEADING_RE.match(name):
            continue

        if name == "section":
            h = child.find(HEADING_RE, recursive=False)
            iter_blocks(child, extend_path(path, heading_text(h)), out, state)

        elif name == "dl":
            # dl 必须当「容器」展开，不能整块拍平。
            # Sphinx 把每个函数/类写成 <dl class="py function"><dt>签名</dt><dd>说明</dd></dl>，
            # 如果让 dl 掉进下面的 BLOCK_TAGS 分支，整块会被 block_text 压成一段流水，
            # 函数签名就变不成标题了 —— 下面的 dt 分支永远进不去（死代码）。
            # 展开之后，每个函数自成一个 chunk 带自己的签名当标题，这才是 library 页真正的检索单元。
            # 每个 dl 配一份新的 state，避免上一个 dl 的签名窜到下一个 dl 里。
            iter_blocks(child, path, out, {"last": path})

        elif name == "dt":
            dt_text = block_text(child)
            if dt_text and len(dt_text) < 80: # 过滤太长的不当标题
                new_path = extend_path(path, dt_text)
                state["last"] = new_path        # 记下来，给后面的 dd 认领
                out.append((new_path, "text", dt_text + "\n"))
            else:
                state["last"] = path
                out.append((path, "text", dt_text + "\n"))

        elif name == "dd":
            # dd 是 dt 下面那段说明，紧跟在 dt 之后，直接认领 dt 刚存的路径
            dd_text = block_text(child)
            if dd_text and not is_noise(dd_text):
                out.append((state.get("last", path), "text", dd_text + "\n"))

        elif name == "pre":
            text = pre_text(child)
            if text:
                out.append((path, "code", text))

        elif name == "div" and child.find(["p", "pre", "ul", "ol", "dl", "table", "div"]):
            iter_blocks(child, path, out, state)

        elif name in BLOCK_TAGS:
            text = block_text(child)
            if text and not is_noise(text):
                out.append((path, "text", text + "\n"))

# ---------- 各格式 → 统一的 blocks ----------
# 这一段是本文件唯一「和格式有关」的地方。三种格式各有一个抽取函数，
# 但全部返回同一种形状 [(title_path, kind, text)]，后面的打包逻辑完全共用。
# 加一种格式 = 在这里加一个分支，pack()/emit() 一个字都不用改。

def blocks_from_html(path: Path, min_page_chars: int,
                     keep_toc: bool) -> tuple[list[Block], dict, str]:
    """HTML：canonical → profile → section 递归抽块。返回 (blocks, meta, 跳过原因)。"""
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    url = canonical_url(soup)
    if not url:
        return [], {}, "canonical 缺失，无法确认这页是谁"

    profile, known = pick_profile(url)
    title, version = page_meta(soup)
    body = soup.select_one(profile["body"])
    if body is None:
        return [], {}, "正文容器未匹配，需要给这个域名加 profile"

    clean(soup, profile)

    # 注意：clean 会删掉 a.headerlink，那些 ¶ 就在里面。必须在 clean 之后算正文长度，
    # 否则目录页的正文会被 ¶ 撑长，判断失准。
    page_chars = len(body.get_text())
    if page_chars < min_page_chars:
        return [], {}, f"正文 < {min_page_chars} 字符（目录页/存根页）"

    if not keep_toc and looks_like_toc(body):
        return [], {}, "只有链接的目录页"

    blocks: list[tuple[tuple[str, ...], str, str]] = []
    iter_blocks(body, (), blocks)
    if not blocks:
        return [], {}, "抽不出任何内容块"

    meta = {
        "url": url,
        "title": title,
        "version": version,
        "site_known": known,
        "page_chars": page_chars,
    }
    return blocks, meta, ""

def blocks_from_markup(path: Path, fmt: str, url: str,
                       min_page_chars: int) -> tuple[list[Block], dict, str]:
    """rst / md / text：丢给 formats.py，抽出来的块形状和 HTML 完全一致。

    title 传空字符串，让文档自己的第一个一级标题当根 ——
    md/rst 的文件名往往是 index / models 这类无意义的词，而文档里的
    `# 简介`、`Models` 才是真正的主题。没有标题时用路径末段兜底。
    """
    text = decode(path.read_bytes())
    blocks = parse_markup(text, fmt, "")

    # 没有标题的文档：path 全是空元组，用文件名补上，否则 title_path 会是空串
    if blocks and all(not p for p, _, _ in blocks):
        stem = path.stem or "untitled"
        blocks = [((stem,), k, t) for _, k, t in blocks]

    page_chars = sum(len(t) for _, _, t in blocks)
    if page_chars < min_page_chars:
        return [], {}, f"正文 < {min_page_chars} 字符"

    # 版本号对 md/rst 没有 <title> 可取，从正文里找「Python 3.x 文档」这种字样。
    # 找不到就留空，不要瞎猜。
    m = VERSION_RE.search(text[:4000])
    meta = {
        "url": url,
        "title": blocks[0][0][0] if blocks and blocks[0][0] else path.stem,
        "version": m.group(1) if m else "",
        "site_known": True,          # 源码格式没有 profile 概念，谈不上「未登记」
        "page_chars": page_chars,
    }
    return blocks, meta, ""

def url_from_layout(path: Path, raw_dir: Path) -> str:
    """按 raw 的目录约定反推源 URL。真正的规则在 layout.py，这里只做路径换算。"""
    try:
        return url_of_rel(path.relative_to(raw_dir).as_posix())
    except ValueError:
        return ""

# ---------- 打包成 chunk-------------

def split_sentences(text: str) -> list[str]:
    """按句子分割文本。"""
    parts = [s.strip() for s in SENT_SPLIT_RE.split(text) if s and s.strip()]
    parts = parts or [text]
    # SENT_SPLIT_RE 里有 \n+ 这个分支，会把块尾那个换行当分隔符吃掉。
    # 换行是块与块的边界标记（iter_blocks 里每块末尾都补了一个），
    # 这里必须补回去，否则 "Lib/x.py" 和下一段英文会粘成 "Lib/x.pyThe operator..."
    if text.endswith("\n") and parts:
        parts[-1] = parts[-1] + "\n"
    return parts

def split_code(code: str, limit: int) -> list[str]:
    """按行分割代码。"""
    if len(code) <= limit:
        return [code]
    out, buf, size = [], [], 0
    for ln in code.split("\n"):
        if size + len(ln) + 1 > limit and buf:
            out.append("\n".join(buf))
            buf, size = [], 0
        buf.append(ln)
        size += len(ln) + 1
    if buf:
        out.append("\n".join(buf))
    return out

def hard_split(text: str, limit: int, overlap: int) -> list[str]:
    """按长度分割文本。"""
    if len(text) <= limit:
        return [text]
    step = max(limit - overlap, 1)
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + limit])
        if i + limit >= len(text):
            break
        i += step
    return out

def tail_overlap(pieces: list[tuple[str, str]], limit: int) -> list[tuple[str, str]]:
    """尾部重叠。"""
    tail, acc = [], 0
    for kind, text in reversed(pieces):
        if acc + len(text) > limit:
            break
        tail.insert(0, (kind, text))
        acc += len(text)
    return tail

def pack(pieces: list[tuple[str, str]]) -> list[str]:
    """打包成多个 chunk。"""
    chunks: list[str] = []
    cur: list[tuple[str, str]] = []
    size = 0

    def flush() -> None:
        nonlocal cur, size
        if not cur:
            return
        buf = []
        for kind, text in cur:
            buf.append("\n" + text + "\n" if kind == "code" else text)
        body = re.sub(r"\n{3,}", "\n\n", "".join(buf).strip())
        if body:
            chunks.append(body)
        cur = tail_overlap(cur, CHUNK_OVERLAP)
        size = sum(len(t) for _, t in cur)

    for kind, text in pieces:
        if kind == "code":
            units = [("code", p) for p in split_code(text, CODE_MAX)]
        else:
            units = [("text", s) for s in split_sentences(text)]
        
        for k, u in units:
            if size + len(u) > CHUNK_MAX and cur:
                flush()
                if size + len(u) > CHUNK_MAX:
                    cur, size = [], 0
            if len(u) > CHUNK_MAX:
                chunks.extend(hard_split(u, CHUNK_MAX, CHUNK_OVERLAP))
                continue
            cur.append((k, u))
            size += len(u)
    flush()
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]

# ---------------- 主流程---------------

def chunks_from_blocks(blocks: list[Block], meta: dict, fmt: str,
                       source_path: str) -> list[dict]:
    """把 blocks 按标题路径分组、打包成 chunk，并补上所有元数据字段。

    分组的意义：同一个 title_path 下的段落和代码块，是同一个检索单元。
    分开打包的话，「函数签名 + 它的说明」会被切成两半。
    """
    url = meta["url"]
    key = doc_key(url)
    section = section_of(url)
    site = site_of(url)
    page_title = meta.get("title", "")
    version = meta.get("version", "")
    known = meta.get("site_known", True)

    chunks: list[dict] = []
    group_path, group_pieces = None, []

    def emit(p: tuple[str, ...], pieces: list[tuple[str, str]]) -> None:
        if not pieces:
            return
        path_str = " > ".join(x for x in p if x) or page_title
        for text in pack(pieces):
            chunks.append({
                "chunk_id": f"{key}-{len(chunks):03d}",
                "doc_id": key,
                "source_file": message_path(source_path),
                "source_path": source_path,
                "format": fmt,
                "site": site,
                "section": section,
                "site_known": known,
                "url": url,
                "page_title": page_title,
                "title_path": path_str,
                "text": text,
                "char_len": len(text),
                "has_code": bool(re.search(r"(?m)^\s{4,}\S", text)),
                "version": version,
                "_digest": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
            })

    for p, kind, text in blocks:
        if p != group_path and group_pieces:
            emit(group_path, group_pieces)
            group_pieces = []
        group_path = p
        group_pieces.append((kind, text))
    emit(group_path, group_pieces)

    seen, uniq = set(), []
    for c in chunks:
        if c["_digest"] in seen:
            continue
        seen.add(c["_digest"])
        uniq.append(c)
    for c in uniq:
        c.pop("_digest", None)
    return uniq

def message_path(source_path: str) -> str:
    """source_file 只保留文件名，方便在报错里指认是哪一篇。"""
    return source_path.rsplit("/", 1)[-1]

def parse_one(path: Path, raw_dir: Path, manifest: dict[str, str],
              min_page_chars: int = MIN_PAGE_CHARS,
              keep_toc: bool = False) -> tuple[list[dict] | None, dict, str]:
    ''' 解析一个文件

    返回 (chunk 列表, meta, 跳过原因)。
    chunk 列表为 None 时，第三个值是跳过原因，供主流程汇总打印。
    '''
    rel = path.relative_to(raw_dir).as_posix()
    sniffed = detect_format(path.read_bytes())

    # raw 的第一层目录（格式）是抓取时确定的，比嗅探权威。
    # 嗅探的弱结论是 text —— 那只是「没找到强特征」，不代表内容不是 rst：
    # Django 的 howto/delete-app 只有标题加三段话，一个指令都没有，嗅探必然给 text。
    # 但嗅探出「明确的另一种格式」时要推翻目录层，那是真的存错了。
    layer = rel.split("/")[0]
    fmt = layer if (layer in FORMAT_PRIORITY and sniffed == "text") else sniffed
    if fmt == "text" and layer == "text":
        fmt = "text"

    if fmt == "html":
        blocks, meta, why = blocks_from_html(path, min_page_chars, keep_toc)
        if not blocks:
            return None, {}, why
        # 展示地址：优先 urls.txt 里那一条（带语言前缀），退回 canonical 补前缀
        url = manifest.get(rel) or localized_url(meta["url"])
    else:
        # rst / md / text 没有 canonical，URL 只能靠 manifest 或目录结构
        url = manifest.get(rel) or url_from_layout(path, raw_dir)
        if not url:
            return None, {}, "推不出源 URL（目录结构不符合 格式/域名/路径）"
        blocks, meta, why = blocks_from_markup(path, fmt, url, min_page_chars)
        if not blocks:
            return None, {}, why

    meta["url"] = url
    chunks = chunks_from_blocks(blocks, meta, fmt, rel)
    if not chunks:
        return None, meta, "打包后没有超过长度下限的 chunk"
    return chunks, meta, ""

def main() -> None:
    ''' 主函数 '''
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只切前 N 篇，验收用")
    ap.add_argument("--out", default=str(OUT_FILE), help="输出路径")
    ap.add_argument("--prefix", default=None, help="只收路径含该前缀的页，如 /library/")
    ap.add_argument("--section", default=None, help="只收某个栏目，如 library / tutorial")
    ap.add_argument("--format", dest="fmt", default=None,
                    help="只处理某种格式：html / rst / md / text")
    ap.add_argument("--min-page-chars", type=int, default=MIN_PAGE_CHARS,
                    help=f"整页正文低于此值丢弃，默认 {MIN_PAGE_CHARS}")
    ap.add_argument("--keep-toc", action="store_true", help="保留只有链接的目录页")
    ap.add_argument("--report", default=None,
                    help="把本次统计写成 JSON（篇数/chunk/长度分位/各站各栏目/丢弃原因）")
    args = ap.parse_args()

    # 递归扫，不再限定 *.html —— raw 现在是「格式/域名/路径」三层结构
    files = sorted(p for p in RAW_DIR.rglob("*") if p.is_file())
    if args.fmt:
        # 按格式筛的时候不能只看扩展名（Django 的 .txt 是 rst），要先嗅探再筛
        files = [p for p in files if detect_format(p.read_bytes()) == args.fmt]
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit("data/raw/ 下没有可解析的文件，先跑 crawl.py")

    manifest = load_manifest()
    expected = set(read_urls())

    # 先全部解析，再决定同一篇文档留哪份格式。
    # 不边解析边去重，是因为文件按路径排序，html/ 排在 rst/ 前面 ——
    # 先到的会是渲染页，后到的才是更干净的源码，边处理边留就留反了。
    results: list[tuple[str, int, list[dict], dict]] = []
    skipped: list[tuple[str, str, str]] = []
    unknowns: dict[str, int] = {}
    not_in_urls = 0

    for i, path in enumerate(files):
        rel = path.relative_to(RAW_DIR).as_posix()
        chunks, meta, why = parse_one(path, RAW_DIR, manifest, args.min_page_chars, args.keep_toc)
        if not chunks:
            skipped.append((rel, meta.get("url", ""), why))
            continue

        url = meta["url"]
        prefix = args.prefix
        if prefix and prefix not in urlparse(url).path:
            skipped.append((rel, url, f"不在前缀 {prefix} 内"))
            continue
        if args.section and chunks[0]["section"] != args.section:
            skipped.append((rel, url, f"栏目是 {chunks[0]['section'] or '根'}，不是 {args.section}"))
            continue
        if expected and url not in expected:
            not_in_urls += 1

        if not chunks[0]["site_known"]:
            host = site_of(url)
            unknowns[host] = unknowns.get(host, 0) + 1

        results.append((doc_key(url), FORMAT_PRIORITY.get(chunks[0]["format"], 0), chunks, meta))

        if (i + 1) % 50 == 0:
            print(f"  ... 已处理 {i + 1}/{len(files)} 篇")

    # 同一篇文档只留优先级最高的一份：rst > md > html > text
    best: dict[str, tuple[int, list[dict]]] = {}
    replaced: list[tuple[str, str, str]] = []
    for key, prio, chunks, _ in results:
        fmt = chunks[0]["format"]
        if key in best and best[key][0] >= prio:
            replaced.append((key, fmt, best[key][1][0]["format"]))
            continue
        if key in best:
            replaced.append((key, best[key][1][0]["format"], fmt))
        best[key] = (prio, chunks)

    kept: list[dict] = []
    for _, (_, chunks) in best.items():
        kept.extend(chunks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in kept:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    docs = len({c["doc_id"] for c in kept})
    lens = sorted(c["char_len"] for c in kept)
    total_chars = sum(lens)
    code_chunks = sum(1 for c in kept if c["has_code"])
    too_short = sum(1 for c in kept if c["char_len"] < 120)

    def quantile(q: float) -> int:
        """按位置取分位数。语料只有一万多块，插值那点精度没有意义。"""
        return lens[min(len(lens) - 1, int(len(lens) * q))] if lens else 0

    print(f"\n完成：{docs} 篇 → {len(kept)} 个 chunk，写入 {out}")
    if lens:
        print(f"字符数：平均 {total_chars // len(lens)}，中位 {lens[len(lens) // 2]}，"
              f"p90 {quantile(0.9)}，max {lens[-1]}")
        print(f"含代码的 chunk：{code_chunks} 个（{code_chunks * 100 // len(lens)}%）")
        print(f"短于 120 字符的 chunk：{too_short} 个")

    # 按格式汇总：这是「加了新格式到底进没进语料」的直接证据
    by_fmt: dict[str, int] = {}
    for c in kept:
        by_fmt[c["format"]] = by_fmt.get(c["format"], 0) + 1
    print("\n按格式：" + "，".join(f"{k} {v} chunk" for k, v in sorted(by_fmt.items())))

    # 按域名汇总：分域建索引之前的规模预检
    by_site: dict[str, list[int]] = {}
    for c in kept:
        by_site.setdefault(c["site"], []).append(c["char_len"])
    if len(by_site) > 1:
        print("\n按域名：")
        for s, ls in sorted(by_site.items(), key=lambda kv: -len(kv[1])):
            n = len({c["doc_id"] for c in kept if c["site"] == s})
            print(f"  {s:<26} {n:>4} 篇  {len(ls):>5} chunk  中位 {sorted(ls)[len(ls) // 2]} 字")

    # 按栏目汇总：以后加语料时，一眼能看出各栏目各占多少，混没混进来别的东西
    by_section: dict[str, list[int]] = {}
    for c in kept:
        by_section.setdefault(c["section"] or "(根)", []).append(c["char_len"])
    if len(by_section) > 1 or "(根)" not in by_section:
        print("\n按栏目：")
        for sec, lens_ in sorted(by_section.items(), key=lambda kv: -len(kv[1])):
            docs_n = len({c["doc_id"] for c in kept if (c["section"] or "(根)") == sec})
            print(f"  {sec:<12} {docs_n:>4} 篇  {len(lens_):>5} chunk  中位 {sorted(lens_)[len(lens_) // 2]} 字")

    # 丢弃原因先归组，打印和写报告共用这一份，避免两处口径不一致
    skipped_by_reason: dict[str, list[tuple[str, str]]] = {}
    for rel, u, why in skipped:
        skipped_by_reason.setdefault(why, []).append((rel, u))

    print(f"\n跳过 {len(skipped)} 篇")
    if skipped_by_reason:
        # 按原因分组，比逐条列 260 行有用得多
        for why, items in sorted(skipped_by_reason.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(items):>3} 篇  {why}")
            for rel, u in items[:4]:
                print(f"        {rel}")
            if len(items) > 4:
                print(f"        ... 还有 {len(items) - 4} 篇")

    # 双份文档被替换掉的记录。这条数必须能被解释清楚：
    # 每一对都是「同一篇的 html 和 rst」，留下的是优先级高的那份
    if replaced:
        print(f"\n同一篇存在多种格式，按优先级取舍 {len(replaced)} 次（留 rst > md > html）")

    if unknowns:
        print("\n[提示] 以下域名没有登记 profile，用的是通用选择器，建议补一条：")
        for host, n in unknowns.items():
            print(f"  {host}  {n} 篇")

    if not_in_urls:
        print(f"\n[提示] {not_in_urls} 篇的 URL 不在 data/urls.txt 里 —— "
              f"可能是直接放进去的样本，也可能是 urls.txt 没跟着更新。")

    # ---------- 统计报告 ----------
    # 屏幕上的数字关掉终端就没了，报告落到盘上才算证据。
    # 后面「改了切分参数有没有变好」靠它对比，不用重跑再靠记忆。
    if args.report:
        # 语料指纹：光看篇数和 chunk 数不够，改了排序或字段也会「数字一样、内容不同」。
        # 有这个哈希，重跑之后一句「一样 / 不一样」就有依据。
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        report = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "corpus_file": out.as_posix(),
            "corpus_sha256": digest,
            "params": {
                "chunk_max": CHUNK_MAX,
                "chunk_overlap": CHUNK_OVERLAP,
                "min_page_chars": args.min_page_chars,
                "min_chunk_chars": MIN_CHUNK_CHARS,
                "code_max": CODE_MAX,
                "keep_toc": bool(args.keep_toc),
            },
            "files_scanned": len(files),
            "docs": docs,
            "chunks": len(kept),
            "chars_total": total_chars,
            "chars": {
                "avg": total_chars // len(lens) if lens else 0,
                "median": quantile(0.5),
                "p90": quantile(0.9),
                "max": lens[-1] if lens else 0,
            },
            "code_chunks": code_chunks,
            "code_pct": code_chunks * 100 // len(lens) if lens else 0,
            "short_lt120": too_short,
            "short_pct": too_short * 100 // len(lens) if lens else 0,
            "by_format": dict(sorted(by_fmt.items())),
            "by_site": {
                s: {
                    "docs": len({c["doc_id"] for c in kept if c["site"] == s}),
                    "chunks": len(ls),
                    "median": sorted(ls)[len(ls) // 2],
                }
                for s, ls in sorted(by_site.items(), key=lambda kv: -len(kv[1]))
            },
            "by_section": {
                (sec or "(根)"): {
                    "docs": len({c["doc_id"] for c in kept if (c["section"] or "(根)") == sec}),
                    "chunks": len(ls),
                    "median": sorted(ls)[len(ls) // 2],
                }
                for sec, ls in sorted(by_section.items(), key=lambda kv: -len(kv[1]))
            },
            "skipped_total": len(skipped),
            "skipped_by_reason": {why: len(items) for why, items in skipped_by_reason.items()},
            "skipped": [{"path": rel, "url": u, "reason": why} for rel, u, why in skipped],
            "replaced_by_format": len(replaced),
            "unknown_hosts": unknowns,
            "urls_list_total": len(expected),
            "urls_not_in_list": not_in_urls,
        }
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n统计报告：{rp}")


if __name__ == "__main__":
    main()
