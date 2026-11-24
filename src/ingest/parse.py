"""语料清洗与切分：data/raw/*.html → data/corpus.jsonl，一行一个 chunk。

为什么不一一行一篇文档：
    检索的最小单位是 chunk，不是文档。一行一篇的话，第 3 步的重排、
    第 6 步的引用溯源全部做不了，等于推倒重来。

为什么按 <section> 递归：
    Sphinx 用 <section id=...> 把「标题 + 该标题下的内容」包成一个整体，
    嵌套 section 就是天然的标题树。递归进去就拿到了 title_path。

URL 的来源：
    身份校验用页面里的 <link rel="canonical">，但展示地址用 urls.txt 里的。
    因为 canonical 会丢掉 /zh-cn 语言前缀。

扫描范围：
    扫 data/raw/ 下所有 *.html，不预设站点、不预设栏目。以后往 raw 里丢新站的
    页面也能直接跑：域名不在 SITE_PROFILES 里就用 DEFAULT_PROFILE 兜底，
    每个 chunk 都会带上 section 字段（从 URL 路径推断的栏目名），
    要分域检索时用 --section 过滤，不用改代码。

用法：
    python -m src.ingest.parse                    # 扫全部，不筛栏目
    python -m src.ingest.parse --limit 10         # 只切前 10 篇，验收用
    python -m src.ingest.parse --section library  # 只要 library 栏目
    python -m src.ingest.parse --prefix /library/ # 按路径前缀筛（旧参数，保留）
    python -m src.ingest.parse --keep-toc         # 不丢弃目录页
"""

from __future__ import annotations #导入未来定义的类,作用：允许在类定义之前使用类名
import argparse   #导入命令行解析模块
import hashlib    #导入哈希模块用于生成文档的哈希值
import json       #导入json模块用于处理json文件
import re         #导入正则表达式模块
import sys        #导入系统模块
from pathlib import Path #导入路径模块
from urllib.parse import urlparse #导入urlparse模块用于解析url

from bs4 import BeautifulSoup #导入BeautifulSoup模块用于解析html文件

#解释下列变量：ROOT：项目根目录，RAW_DIR：原始数据目录，URLS_FILE：URL文件路径，OUT_FILE：输出文件路径
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
URLS_FILE = ROOT / "data" / "urls.txt"
OUT_FILE = ROOT / "data" / "corpus.jsonl"

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

# ---------- 正则 ----------
#解释下列变量：HEADING_RE：标题正则表达式，SENT_SPLIT_RE：句子分割正则表达式，VERSION_RE：版本号正则表达式
HEADING_RE = re.compile(r"^h[1-6]$")               
SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])|(?<=[.!?])\s+|\n+")
VERSION_RE = re.compile(r"Python\s+([\d.]+)")
# 路径里的语言代码段（zh-cn / en-us / ja）与版号段（3 / 3.15 / v2），推断栏目名时跳过
LANG_SEG_RE = re.compile(r"^[a-z]{2}(-[a-z]{2})?$")
VER_SEG_RE = re.compile(r"^v?\d+(\.\d+)*$")

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
    while parts and (LANG_SEG_RE.match(parts[0]) or VER_SEG_RE.match(parts[0])):
        parts = parts[1:]
    return parts[0] if parts else ""

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

def parse_one(path: Path, min_page_chars: int = MIN_PAGE_CHARS,
              keep_toc: bool = False) -> tuple[list[dict] | None, str, str]:
    ''' 解析一个文件

    返回 (chunk 列表, canonical_url, 跳过原因)。
    chunk 列表为 None 时，第三个值是跳过原因，供主流程汇总打印。
    '''
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    url = canonical_url(soup)
    if not url:
        return None, "", "canonical 缺失，无法确认这页是谁"

    profile, known = pick_profile(url)
    title, version = page_meta(soup)
    body = soup.select_one(profile["body"])
    if body is None:
        return None, url, "正文容器未匹配，需要给这个域名加 profile"

    clean(soup, profile)

    # 注意：clean 会删掉 a.headerlink，那些 ¶ 就在里面。必须在 clean 之后算正文长度，
    # 否则目录页的正文会被 ¶ 撑长，判断失准。
    if len(body.get_text()) < min_page_chars:
        return None, url, f"正文 < {min_page_chars} 字符（目录页/存根页）"

    if not keep_toc and looks_like_toc(body):
        return None, url, "只有链接的目录页"

    blocks: list[tuple[tuple[str, ...], str, str]] = []
    iter_blocks(body, (), blocks)
    if not blocks:
        return None, url, "抽不出任何内容块"

    doc_id = path.stem
    section = section_of(url)
    chunks: list[dict] = []
    group_path, group_pieces = None, []

    def emit(p: tuple[str, ...], pieces: list[tuple[str, str]]) -> None:
        if not pieces:
            return
        path_str = " > ".join(x for x in p if x) or title
        for text in pack(pieces):
            chunks.append({
                "chunk_id": f"{doc_id}-{len(chunks):03d}",
                "doc_id": doc_id,
                "source_file": path.name,
                "section": section,
                "site_known": known,
                "url": url,
                "page_title": title,
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
    return uniq, url, ""

def main() -> None:
    ''' 主函数 '''
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只切前 N 篇，验收用")
    ap.add_argument("--out", default=str(OUT_FILE), help="输出路径")
    ap.add_argument("--prefix", default=None, help="只收路径含该前缀的页，如 /library/")
    ap.add_argument("--section", default=None, help="只收某个栏目，如 library / tutorial")
    ap.add_argument("--min-page-chars", type=int, default=MIN_PAGE_CHARS,
                    help=f"整页正文低于此值丢弃，默认 {MIN_PAGE_CHARS}")
    ap.add_argument("--keep-toc", action="store_true", help="保留只有链接的目录页")
    args = ap.parse_args()

    files = sorted(RAW_DIR.glob("*.html"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit("data/raw/ 下没有 HTML，先跑 crawl.py")

    expected = read_urls()
    kept, skipped, mismatch = [], [], []
    seen_urls: dict[str, str] = {}   # canonical_url → 第一次出现的 doc_id，用来跨文件去重
    unknowns: dict[str, int] = {}    # 未登记域名 → 篇数
    total_chars, code_chunks, too_short = 0, 0, 0

    for i, path in enumerate(files):
        idx = int(path.stem) if path.stem.isdigit() else 0
        guess = expected[idx - 1] if 0 < idx <= len(expected) else ""

        chunks, url, why = parse_one(path, args.min_page_chars, args.keep_toc)
        if not chunks:
            skipped.append((path.stem, url or guess, why))
            continue

        prefix = args.prefix
        if prefix and prefix not in urlparse(url).path:
            skipped.append((path.stem, url, f"不在前缀 {prefix} 内"))
            continue
        if args.section and chunks[0]["section"] != args.section:
            skipped.append((path.stem, url, f"栏目是 {chunks[0]['section'] or '根'}，不是 {args.section}"))
            continue

        # 跨文件去重：同一个 canonical URL 被抓了两次（重爬、编号变动都会造成），
        # 只保留第一份，后面的直接丢。判断依据是 URL 而不是文件名 —— 文件名会变，URL 不会。
        if url in seen_urls:
            skipped.append((path.stem, url, f"重复 URL，内容已在 {seen_urls[url]}"))
            continue
        seen_urls[url] = path.stem

        if not chunks[0]["site_known"]:
            host = urlparse(url).netloc
            unknowns[host] = unknowns.get(host, 0) + 1

        if guess and basename(guess) != basename(url):
            mismatch.append((path.stem, guess, url))
            # 编号对不上，说明 urls.txt 那个位置已经不是这一页了，不能拿它当展示地址。
            # 退回 canonical，但要把语言前缀补回去，否则引用会跳到英文页。
            fixed = localized_url(url)
            for c in chunks:
                c["url"] = fixed
        elif guess:
            for c in chunks:
                c["url"] = guess
        else:
            fixed = localized_url(url)
            for c in chunks:
                c["url"] = fixed

        for c in chunks:
            total_chars += c["char_len"]
            code_chunks += 1 if c["has_code"] else 0
            too_short += 1 if c["char_len"] < 120 else 0
        kept.extend(chunks)
        
        if (i + 1) % 50 == 0:
            print(f"  ... 已处理 {i + 1}/{len(files)} 篇，累计 {len(kept)} chunk")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in kept:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    docs = len({c["doc_id"] for c in kept})
    lens = sorted(c["char_len"] for c in kept)
    print(f"\n完成：{docs} 篇 → {len(kept)} 个 chunk，写入 {out}")
    if lens:
        print(f"字符数：平均 {total_chars // len(lens)}，中位 {lens[len(lens) // 2]}，"
              f"p90 {lens[int(len(lens) * 0.9)]}，max {lens[-1]}")
        print(f"含代码的 chunk：{code_chunks} 个（{code_chunks * 100 // len(lens)}%）")
        print(f"短于 120 字符的 chunk：{too_short} 个")

    # 按栏目汇总：以后加语料时，一眼能看出各栏目各占多少，混没混进来别的东西
    by_section: dict[str, list[int]] = {}
    for c in kept:
        by_section.setdefault(c["section"] or "(根)", []).append(c["char_len"])
    if len(by_section) > 1 or "(根)" not in by_section:
        print("\n按栏目：")
        for sec, lens_ in sorted(by_section.items(), key=lambda kv: -len(kv[1])):
            docs_n = len({c["doc_id"] for c in kept if (c["section"] or "(根)") == sec})
            print(f"  {sec:<12} {docs_n:>4} 篇  {len(lens_):>5} chunk  中位 {sorted(lens_)[len(lens_) // 2]} 字")
    
    print(f"跳过 {len(skipped)} 篇")
    if skipped:
        # 按原因分组，比逐条列 260 行有用得多
        reasons: dict[str, list[tuple[str, str]]] = {}
        for doc_id, u, why in skipped:
            reasons.setdefault(why, []).append((doc_id, u))
        for why, items in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(items):>3} 篇  {why}")
            for doc_id, u in items[:4]:
                print(f"        {doc_id} {basename(u) or '?'}")
            if len(items) > 4:
                print(f"        ... 还有 {len(items) - 4} 篇")

    if unknowns:
        print("\n[提示] 以下域名没有登记 profile，用的是通用选择器，建议补一条：")
        for host, n in unknowns.items():
            print(f"  {host}  {n} 篇")

    if mismatch:
        print(f"\n[警告] {len(mismatch)} 篇的编号与页面自称的 URL 对不上：")
        for doc_id, guess, real in mismatch[:8]:
            print(f"  {doc_id}  urls.txt 期望 {basename(guess)}，页面实际是 {basename(real)}")
        print("  说明 raw 里有错位文件：编号由列表位置决定，重爬后位置一变就错。")
        print("  不影响切分（身份以 canonical 为准），但补语料前建议先把 crawl.py 的编号绑到 URL 上。")

if __name__ == "__main__":
    main()
