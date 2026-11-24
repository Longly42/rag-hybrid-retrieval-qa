"""多种文本格式的探测与解析：HTML 之外，还要能吃 rst / md / 纯文本。

为什么单独一个文件：
    parse.py 里那套 HTML 逻辑（section 递归、dl/dt/dd、headerlink、span.go）是 Sphinx
    特有的，rst 和 md 的结构完全不同。塞在一起会变成一千多行的 if-else 坟场。
    这里只做一件事：把任意格式变成同一种中间形态 ——

        [(title_path, "text" | "code", 文本), ...]

    然后交给 parse.py 原有的切分与打包。加一种格式 = 加一个函数 + 注册一行，
    切分、打包、chunk schema 一个字都不用改。这就是「健壮」的实际含义。

为什么格式靠内容嗅探、不靠扩展名：
    两个实测反例：
      · Django 的 _sources 源文件扩展名是 .txt，内容其实是 rst
      · CS-Notes 的 README.md 里混着 <div align="center"> 标签
    按扩展名分流，第一天就分错。

对外只暴露三个函数：
    decode(raw)                      字节 → 文本，猜编码
    detect_format(raw)               内容 → "html" / "rst" / "md" / "text"
    parse_markup(text, fmt, title)   文本 + 格式 → blocks
"""

from __future__ import annotations

import re

# 统一的中间形态：每个块是 (标题路径, 类型, 文本)
# 类型只有两种，和 parse.py 里 iter_blocks() 的输出完全一致
# "text" = 正文段落，"code" = 代码块（打包时不能被断句、不能被字符数硬切）
Block = tuple[tuple[str, ...], str, str]


# ============ 编码 ============

def decode(raw: bytes) -> str:
    """把字节解成文本。

    顺序有讲究：utf-8 优先，因为 Sphinx/Django 的源文件基本都是 utf-8；
    gb18030 兜底，能覆盖 GBK/GB2312（它是超集），老的中文 md 仓常是 GBK；
    最后 replace 强制解码，宁可留几个乱码字符，也不能因为一个坏字节整篇丢掉。
    """
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ============ 格式探测 ============

# 纯装饰行：一整行都是同一个标点，长度 >= 3。rst 用它画标题上下划线。
# 只认这些符号，不收 "." 和 "_" —— 它们会被省略号和 snake_case 误命中。
RST_ADORN_RE = re.compile(r"^([=\-~^\"'*+#])\1{2,}\s*$")


def detect_format(raw: bytes) -> str:
    """看内容判断格式。前 8 KB 足够，格式特征都在开头。

    判定顺序 = 特征强度从强到弱，先命中先返回：
      1. HTML 结构标签（<!DOCTYPE html / <html / </body>）—— 几乎不会误判
      2. rst 指令行（.. xxx::）—— rst 独有
      3. md 的 ATX 标题（# ）或代码围栏（```）—— 先于 rst 下划线判断，
         因为 md 也支持 setext 标题（文字下一行 =====），那和 rst 长得一样
      4. rst 装饰行出现 3 次以上 —— Sphinx 源文件的典型形态
      5. 都不是 → 纯文本兜底
    """
    head = decode(raw[:8192])
    low = head.lower()

    # HTML 只在**文件开头**认。真正的 HTML 文档，doctype 一定在头几百字节里；
    # 而 rst/md 讲 Web 的时候正文里常贴 <!DOCTYPE html> 示例，
    # 只要出现就当 html，会把一整篇 Django 文档判错（实测 topics/http/views 就这么丢的）。
    # 不拿 </body> 当判据：它几乎总在代码示例里出现，而且真 HTML 必有 doctype 或 <html>。
    head_low = low[:2048]
    if re.search(r"(?m)^\s{0,3}<(!doctype\s+html|html[\s>])", head_low):
        return "html"

    if re.search(r"(?m)^\.\.\s+[\w:+.-]+::", head):
        return "rst"

    if re.search(r"(?m)^#{1,6}\s+\S", head) or re.search(r"(?m)^(```|~~~)", head):
        return "md"

    # 上下划线标题最少就是两行（标题上面一行、下面一行）。
    # 阈值定在 3 会把只有一两个小节的短文判成纯文本 —— howto/delete-app 就一篇标题加三段话。
    adorns = [ln for ln in head.splitlines() if RST_ADORN_RE.match(ln)]
    if len(adorns) >= 2:
        return "rst"

    return "text"


# ============ 行内标记清洗 ============
# 这一节的目的只有一个：把「标记」还原成「词」。
# 不还原的话，BM25 的倒排表里会出现 `:` `func:` `\`` 这种碎片，
# 而 os.path.join 会被切成 "func" "os.path.join" 两个 token —— 字面匹配直接废掉。
# 这和 HTML 那边 unwrap inline 标签是同一件事，只是换了套语法。

# :func:`os.path.join` / :class:`~collections.OrderedDict`
RST_ROLE_RE = re.compile(r":[a-zA-Z][\w:+-]*:`([^`]+)`")
# `显示文字 <https://url>`_  —— 带链接目标的引用
RST_REF_LINK_RE = re.compile(r"`([^`<]+?)\s*<[^>]*>`_{0,2}")
# `纯文字`_  —— 不带目标的引用
RST_PLAIN_REF_RE = re.compile(r"`([^`]+)`_{1,2}")
# ``字面量``_ —— 双反引号还带下划线的引用形式，少了这条就会残留一个尾巴下划线
RST_LITERAL_REF_RE = re.compile(r"``([^`]+)``_{1,2}")
# **粗体** / *斜体* / ``字面量`` / |替换引用|
RST_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
RST_EM_RE = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")
RST_LITERAL_RE = re.compile(r"``([^`]+)``")
RST_SUBST_RE = re.compile(r"\|([^|\n]+)\|")
# 单反引号引用 `Meta`（不带下划线）。必须排在双反引号之后，
# 否则 ``Person`` 会被拆成两个空引用，中间的文字反而丢了。
RST_SINGLE_REF_RE = re.compile(r"`([^`]+)`")
# 简写引用 fields_ —— rst 允许「单词 + 下划线」直接当引用，没有反引号。
# 要求下划线后面是空白或标点，这样 my_var = 1 里的下划线不会被误伤。
RST_SHORT_REF_RE = re.compile(r"\b(\w+)_(?=[\s.,;:)\]]|$)")
# 脚注引用 [1]_ / [*]_
RST_FOOTNOTE_RE = re.compile(r"\[[^\]]{1,8}\]_")


def _role_text(m: re.Match) -> str:
    """角色标记的内层文字。

    `:djadmin:`manage.py migrate <migrate>`` 这种，尖括号里是链接目标，
    不是要显示的文字。只剥 role 内部的，不在全局做 `<...>` 替换 ——
    全局替换会把代码里的 `a < b` 当成标签吃掉。
    """
    s = m.group(1)
    return re.sub(r"\s*<[^>]*>$", "", s)


def strip_rst_inline(s: str) -> str:
    """剥掉 rst 的行内标记，只留文字。

    顺序不能乱，这是踩过的坑：
      · 先处理 :role:`x`（它里面可能还套着 `text <url>` 的形式）
      · 再处理带目标的引用，然后是单纯引用
      · 双反引号必须排在单反引号之前：反过来的话 ``Person`` 会被
        RST_SINGLE_REF_RE 从中间切开，变成两个空引用，Person 这个词直接消失
      反过来做的话，:func:`os.path.join` 会先被 RST_PLAIN_REF_RE 吃掉反引号，
      剩下一个孤零零的 `:func:` 前缀。
    """
    s = RST_ROLE_RE.sub(_role_text, s)
    s = RST_REF_LINK_RE.sub(lambda m: m.group(1), s)
    s = RST_PLAIN_REF_RE.sub(lambda m: m.group(1), s)
    s = RST_LITERAL_REF_RE.sub(lambda m: m.group(1), s)
    s = RST_LITERAL_RE.sub(lambda m: m.group(1), s)
    s = RST_SINGLE_REF_RE.sub(lambda m: m.group(1), s)
    s = RST_BOLD_RE.sub(lambda m: m.group(1), s)
    s = RST_EM_RE.sub(lambda m: m.group(1), s)
    s = RST_SUBST_RE.sub(lambda m: m.group(1), s)
    s = RST_FOOTNOTE_RE.sub("", s)
    s = RST_SHORT_REF_RE.sub(lambda m: m.group(1), s)
    # 角色标记里的 ~ 前缀表示「不显示完整路径」，例如 :class:`~collections.OrderedDict`
    # 剥完标记后可能剩下孤立的 ~，会污染分词
    s = re.sub(r"(?<![\w.])~(?=[\w])", "", s)
    return re.sub(r"\s+", " ", s).strip()


# ![alt](url) —— 图片，URL 必须丢（否则语料里全是 CDN 链接），alt 有信息价值
MD_IMG_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
# [text](url) —— 链接，留文字
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# `code` —— 行内代码，去反引号
MD_CODE_RE = re.compile(r"`([^`]+)`")
# **粗体** / __粗体__
MD_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
# *斜体* —— 只管星号。下划线斜体不碰，否则 os_path_join 这种标识符会被撕开
MD_EM_RE = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")
# HTML 标签（CS-Notes 那种 md 里混标签）
MD_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][\w-]*(?:\s[^<>]*)?/?>")
# VuePress/VitePress 的 {{ $frontmatter.x }} 插值
MD_MUSTACHE_RE = re.compile(r"\{\{[^}]*\}\}")


def strip_md_inline(s: str) -> str:
    """剥掉 md 的行内标记。顺序同理：先整体结构，再局部强调。

    行内代码必须**先保护起来**再处理其余标记。否则 `` `<Transition>` `` 里的
    尖括号会被当 HTML 标签删掉，只剩下两个孤零零的反引号 ——
    这不是排版问题，是内容直接丢了。
    """
    codes: list[str] = []

    def keep(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    s = MD_CODE_RE.sub(keep, s)
    s = MD_MUSTACHE_RE.sub("", s)
    s = MD_IMG_RE.sub(lambda m: m.group(1), s)     # 图片先于链接，否则 ! 会剩下来
    s = MD_LINK_RE.sub(lambda m: m.group(1), s)
    s = MD_HTML_TAG_RE.sub("", s)
    s = MD_BOLD_RE.sub(lambda m: m.group(2), s)
    s = MD_EM_RE.sub(lambda m: m.group(1), s)
    s = s.replace("\\", "")                        # md 的转义反斜杠
    s = re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], s)
    return re.sub(r"\s+", " ", s).strip()


# ============ 公共小工具 ============

def _need_space(prev: str, nxt: str) -> bool:
    """两个断行之间该不该补空格。

    这段逻辑看着琐碎，但它是「黏连」的唯一防线。rst 和 md 都把一句话折成多行，
    拼回去时的规则是：
      · 英文接英文 → 补空格（contains + the → "contains the"）
      · 中文接中文 → 不补（否则变成「这 是 一 段 话」）
      · 中文接英文 → 补空格（可读性，且不影响分词）
      · 上一行以 :class: 这类角色标记开头 → 必须补，否则 subclasses 和
        django.db.models.Model 会粘成 subclassesdjango.db.models.Model
    """
    if not prev or not nxt:
        return False
    # 下一行以 :role:` 开头 → 一定是 rst 的角色标记，必须隔开，
    # 否则 subclasses 和 django.db.models.Model 会粘成一个词
    if re.match(r"^:[a-zA-Z][\w:+-]*:`", nxt):
        return True
    a, b = prev[-1], nxt[0]
    a_ascii, b_ascii = ord(a) < 128, ord(b) < 128
    # 后置标点紧跟上一行，不空格：句号、逗号、右括号
    post = ".,;!?)]}"
    if a_ascii and b_ascii:
        if b in post:
            return False
        return a.isalnum() or a in post + "\"'*`"
    if not a_ascii and b_ascii:
        # 中文接英文（或英文标点），留个空格。引号也算 ——
        # in 加 "Retrieving 会变成 in"Retrieving，读起来像代码
        return b.isalnum() or b in "\"'([{" 
    return False


def join_lines(lines: list[str]) -> str:
    """把一个段落里的多行拼成一行。规则见 _need_space。"""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if _need_space(out, ln):
            out += " "
        out += ln
    return out


def clean_title(s: str) -> str:
    """标题清洗：剥标记 + 去掉行尾的手工锚点。

    md 里常见 `# 简介 {#introduction}`，花括号那段是锚点，不是标题内容。
    """
    s = strip_rst_inline(strip_md_inline(s))
    s = re.sub(r"\s*\{#[^}]*\}\s*$", "", s)
    s = s.strip(" :：·-—#")
    return s


# ============ rst 解析 ============

# .. directive:: 参数
RST_DIRECTIVE_RE = re.compile(r"^\.\.\s+([\w:+.-]+)::\s*(.*)$")
# 列表项：* / - / + / 1. 后面必须跟空白，否则 *强调* 会被误判成列表
RST_BULLET_RE = re.compile(r"^\s*(?:[*+-]|\d+[.)])\s+\S")

# 纯导航类指令：内容里全是别的页面标题，留下只会污染索引，和 HTML 那边的目录页同理
RST_DROP_DIRECTIVES = {
    "index", "toctree", "module", "currentmodule", "highlight", "default-role",
    "raw", "include", "literalinclude", "image", "figure", "meta", "contents",
    "sectionauthor", "codeauthor", "only", "tabularcolumns", "graphviz",
}

# 代码类指令：后面缩进的内容整块当 code
RST_CODE_DIRECTIVES = {"code-block", "sourcecode", "code", "doctest", "testcode", "ipython"}

# 提示类指令：指令名丢掉，但内容必须留。versionadded 的版本号有检索价值，拼回去。
RST_NOTE_DIRECTIVES = {
    "note", "warning", "seealso", "tip", "important", "caution", "danger",
    "attention", "hint", "error", "admonition", "topic", "sidebar", "example",
    "versionadded", "versionchanged", "deprecated",
}

# 签名类指令：参数部分是函数/类签名，应该当标题用（和 HTML 里 <dt> 的地位一样）
RST_SIG_DIRECTIVES = {
    "function", "class", "method", "attribute", "data", "exception",
    "staticmethod", "classmethod", "decorator", "property",
}


def is_adorn(line: str) -> bool:
    """这一行是不是纯装饰行（rst 的标题下划线 / 上划线）。"""
    return bool(RST_ADORN_RE.match(line))


def gather_indented(lines: list[str], start: int) -> tuple[list[str], int]:
    """从 start 起收集缩进块，返回 (块内文本行, 下一处未消费的下标)。

    规则：
      · 空行先记着，因为块中间允许空行
      · 第一行非空行的缩进决定基准，只收缩进 >= 基准的行
      · 遇到缩进比基准浅的行 → 块结束
      · 文件末尾自然结束

    「基准」这一条是必须的，不能只看「有没有缩进」：
    Django 的列表项下面常写成
        * 第一段（缩进 2）
          ...说明...::
             代码（缩进 4）
          后续段落（缩进 2）
    如果只要缩进就收，代码块会一路吃到「后续段落」，把里面所有的
    :ref:`…` 角色标记当成代码原样留下 —— 实测就是这么漏了 5 处。

    这是 rst 里最重要的一个原语：指令体、代码块、列表续行，全靠它切出来。
    """
    body: list[str] = []
    i, n = start, len(lines)

    # 先跳过开头空行找基准行。基准为 0 说明根本没有缩进块 ——
    # 这时必须原样返回 start，不能把后面所有正文都当成「缩进 >= 0」收进来
    # （第一版就是这么写的，结果 `.. module::` 后面的整篇正文被当指令体丢掉，
    # 一个文件切出 0 个块）。
    j = start
    while j < n and not lines[j].strip():
        j += 1
    if j >= n:
        return [], n
    base = len(lines[j]) - len(lines[j].lstrip())
    if base == 0:
        return [], start

    while i < n:
        ln = lines[i]
        if not ln.strip():
            body.append("")
            i += 1
            continue
        cur = len(ln) - len(ln.lstrip())
        if cur < base:
            break
        body.append(ln)
        i += 1
    # 块尾的空行不属于内容
    while body and not body[-1].strip():
        body.pop()
    return body, i


def dedent(lines: list[str]) -> list[str]:
    """去掉缩进块共同的左侧空白，保留相对缩进（代码块的层级要靠它）。"""
    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    if not indents:
        return lines
    cut = min(indents)
    return [ln[cut:] if len(ln) >= cut else "" for ln in lines]


def gather_list(lines: list[str], start: int) -> tuple[list[str], int]:
    """收集一段连续的 rst 列表，返回 (每一项拼好的一行, 下一处下标)。

    为什么把整个列表合成一块而不是一项一块：
        HTML 那边 <ul> 本来就是整块，一项一个 chunk 会切出几十个十几字的碎片
        （「* 每个模型是一个 Python 类」这种），既召不准又占名额。
        合并之后它是一个完整论据，检索时能一次命中整段语义。

    rst 的列表项可以跨行，续行靠缩进辨认；项与项之间允许有空行。
    """
    items: list[str] = []
    i, n = start, len(lines)
    while i < n:
        if not RST_BULLET_RE.match(lines[i]):
            break
        base = len(lines[i]) - len(lines[i].lstrip())
        buf = [lines[i].strip()]
        i += 1
        # 续行：缩进必须比列表项本身更深。只判「有缩进」会把代码块也吃进来 ——
        # 列表项里的代码比 bullet 深，但和续行一样都是缩进，只能比深度
        while i < n and lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) > base:
            buf.append(lines[i].strip())
            i += 1
        items.append(join_lines(buf))
        # 跳过项间空行，看下一项是不是列表的延续
        j = i
        while j < n and not lines[j].strip():
            j += 1
        if j < n and RST_BULLET_RE.match(lines[j]):
            i = j
            continue
        break
    return items, i


def parse_rst(text: str, title: str = "") -> list[Block]:
    """把 rst 源文件切成块。

    标题层级不靠符号硬编码：
        rst 规定「装饰符第一次出现的顺序」就是层级顺序，不同文档用的符号组合
        完全不一样（有的 `=` `-` `~`，有的只有 `-` `~`）。写死符号=层级，
        换个文件就全错。所以这里维护一张「符号 → 第几级」的表，遇到新符号就追加。
    """
    lines = text.splitlines()
    blocks: list[Block] = []
    heads: list[str] = []      # 当前标题链（不含文档标题）
    order: dict[str, int] = {} # 装饰符 → 层级
    sig = ""                   # 当前函数签名，作用同 HTML 里的 <dt>
    prefix = (title,) if title else ()

    def level_of(sym: str) -> int:
        """符号 → 层级。首次见到就分配下一个层级，之后固定。"""
        if sym not in order:
            order[sym] = len(order)
        return order[sym]

    def cur_path() -> tuple[str, ...]:
        """当前块的标题路径 = 文档标题 + 标题链 + 签名。

        文档标题和第一个章节标题常常是同一个词（Django 的 models.txt 就是这样：
        文件名 Models，第一行标题也是 Models），不去重会得到「Models > Models」，
        白白让 title_path 重复一遍，还会污染 BM25 的重复命中。
        """
        h = list(heads)
        if prefix and h and h[0] == prefix[-1]:
            h = h[1:]
        return prefix + tuple(h) + ((sig,) if sig else ())

    def push(kind: str, body: str) -> None:
        body = body.strip("\n")
        if body.strip():
            blocks.append((cur_path(), kind, body + "\n"))

    def push_code(body: str) -> None:
        # 行首空行是 gather_indented 顺手收进来的，不属于代码内容
        body = body.strip("\n")
        if body.strip():
            blocks.append((cur_path(), "code", body + "\n"))

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]

        # ---- 上划线形式：=== / 标题 / === ----
        if is_adorn(line) and i + 2 < n and lines[i + 1].strip() and is_adorn(lines[i + 2]) \
                and lines[i + 2][0] == line[0]:
            lvl = level_of(line[0])
            heads = heads[:lvl] + [clean_title(lines[i + 1])]
            sig = ""
            i += 3
            continue

        # ---- 下划线形式：标题 / === ----
        if line.strip() and not is_adorn(line) and i + 1 < n and is_adorn(lines[i + 1]):
            lvl = level_of(lines[i + 1][0])
            heads = heads[:lvl] + [clean_title(line)]
            sig = ""
            i += 2
            continue

        # ---- 空行 ----
        if not line.strip():
            i += 1
            continue

        # ---- 指令 ----
        m = RST_DIRECTIVE_RE.match(line)
        if m:
            name = m.group(1).lower()
            arg = m.group(2).strip()
            body, nxt = gather_indented(lines, i + 1)
            nxt = max(nxt, i + 1)   # 没有缩进体时也要保证主循环往前走，不然卡死

            if name in RST_DROP_DIRECTIVES:
                i = nxt
                continue

            if name in RST_CODE_DIRECTIVES:
                # 代码块整体保留，不在这里折行 —— 折行阈值属于 parse.py 的 CODE_MAX
                push_code("\n".join(dedent(body)))
                i = nxt
                continue

            if name in RST_SIG_DIRECTIVES or name.startswith("py:"):
                # 函数/类签名：arg 就是签名本身，挂到标题链上（地位等同 HTML 的 <dt>）
                if arg:
                    sig = clean_title(arg)
                    push("text", sig)          # 签名自己也算一块，保证它一定被看到
                rest = strip_rst_inline(join_lines(dedent(body)))
                if rest:
                    push("text", rest)
                i = nxt
                continue

            if name in RST_NOTE_DIRECTIVES:
                # 提示块：丢掉指令名，保留内容。versionadded 3.8 这种把版本号拼回去，
                # 因为「3.8」是有检索价值的词，丢了就查不到「哪个版本开始有的」
                # arg 必须一起剥标记：admonition 的 arg 是一整句话，里面常带
                # :attr:`~Options.indexes` 这种角色标记，不剥就会原样进语料
                head = f"{name} {strip_rst_inline(arg)}:" if arg else f"{name}:"
                inner = strip_rst_inline(join_lines(dedent(body)))
                push("text", f"{head} {inner}" if inner else head)
                i = nxt
                continue

            # 未登记的指令：一律当提示块处理，宁可多留正文，不要静默丢内容
            inner = strip_rst_inline(join_lines(dedent(body)))
            if inner:
                push("text", inner)
            i = nxt
            continue

        # ---- .. 开头的注释（没有 :: 的那种） ----
        if line.startswith(".."):
            # 单行注释 `.. 这是注释` 没缩进体；带缩进体的顺手吃掉，反正都是注释
            _, nxt = gather_indented(lines, i + 1)
            i = max(nxt, i + 1)
            continue

        # ---- 列表 ----
        if RST_BULLET_RE.match(line):
            items, nxt = gather_list(lines, i)
            # 每一项单独剥标记，不能整块剥 —— strip_rst_inline 会把 \s+ 压成空格，
            # 整块剥就把项与项之间的换行也压平了，列表退化成一句话
            txt = "\n".join(strip_rst_inline(x) for x in items if x.strip())
            push("text", txt)
            i = nxt
            continue

        # ---- 普通段落 ----
        # 这里必须处理 `::`：它可能出现在段落中间，不一定在段首。
        # Django 的 models.txt 第 25-26 行是
        #     This example model defines a ``Person``, which has a ``first_name`` and
        #     ``last_name``::
        # `::` 在第二行行尾。只在段首查 `::`，整段后面的缩进代码就会被当成正文
        # 收进段落，结果 class 定义被压成一行流水，代码块彻底丢失。
        para: list[str] = []
        expect_code = False
        while i < n:
            ln = lines[i]
            if not ln.strip():
                break
            if is_adorn(ln) or RST_DIRECTIVE_RE.match(ln) or ln.startswith(".."):
                break
            # 下一行是装饰行 → 当前行其实是标题，交回上面处理
            if i + 1 < n and is_adorn(lines[i + 1]):
                break
            # 段落中途冒出列表项 → 列表到此为止，交回主循环
            if RST_BULLET_RE.match(ln):
                break
            if ln.rstrip().endswith("::"):
                lead = ln.rstrip()[:-2]
                if lead.strip():
                    para.append(lead)
                expect_code = True
                i += 1
                break
            para.append(ln)
            i += 1

        if para:
            txt = strip_rst_inline(join_lines(para))
            if txt:
                push("text", txt)

        if expect_code:
            body, nxt = gather_indented(lines, i)
            i = max(nxt, i)
            push_code("\n".join(dedent(body)))
        elif not para:
            i += 1     # 兜底，保证循环一定前进，不会死锁

    return blocks


# ============ md 解析 ============

MD_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
MD_FENCE_RE = re.compile(r"^(\s*)(```+|~~~+)\s*([\w+-]*)\s*$")
# ::: 容器标记。可能单独一行（::: tip），也可能后面直接跟内容（:::info 你正在阅读…）
MD_CONTAINER_RE = re.compile(r"^\s*:::\s*([\w-]*)\s*(.*)$")
MD_TABLE_RE = re.compile(r"^\s*\|")
MD_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# 引用块：行首的 > 是标记，不是内容
MD_QUOTE_RE = re.compile(r"^\s*>\s?")


def parse_md(text: str, title: str = "") -> list[Block]:
    """把 markdown 切成块。

    和 rst 的关键差别：md 的标题层级是显式的（# 的个数就是层级），
    不需要维护符号表。但多了两样东西要处理：frontmatter 和 ::: 容器。
    """
    lines = text.splitlines()

    # ---- frontmatter：文件开头 --- 到下一个 --- 之间全是元数据 ----
    # 不剥掉的话，`footer: false` 会变成标题路径的一部分
    i = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, min(len(lines), 40)):
            if lines[j].strip() == "---":
                # 顺手把 frontmatter 里的 title 捞出来当文档标题
                for ln in lines[1:j]:
                    if ln.lower().startswith("title:") and not title:
                        title = clean_title(ln.split(":", 1)[1])
                i = j + 1
                break

    blocks: list[Block] = []
    heads: list[str] = []
    prefix: tuple[str, ...] = (title,) if title else ()

    n = len(lines)
    while i < n:
        line = lines[i]

        # ---- 代码围栏：整块保留 ----
        m = MD_FENCE_RE.match(line)
        if m:
            fence = m.group(2)[0] * 3
            i += 1
            code: list[str] = []
            while i < n and not lines[i].strip().startswith(fence):
                code.append(lines[i])
                i += 1
            i += 1     # 跳过收尾围栏
            body = "\n".join(code).rstrip()
            if body.strip():
                blocks.append((prefix + tuple(heads), "code", body + "\n"))
            continue

        # ---- ::: info 这类容器标记：丢标记，留内容 ----
        # 有两种形态：单独一行 `::: tip`，或者 `:::info 你正在阅读的是 Vue 3 的文档！`
        # 后者标记后面紧跟正文，整行丢掉会把内容一起丢掉
        m = MD_CONTAINER_RE.match(line)
        if m:
            rest = m.group(2).strip()
            i += 1
            if rest:
                blocks.append((prefix + tuple(heads), "text", strip_md_inline(rest) + "\n"))
            continue

        # ---- 引用块：剥掉行首的 >，整块保留 ----
        if MD_QUOTE_RE.match(line):
            qs: list[str] = []
            while i < n and MD_QUOTE_RE.match(lines[i]):
                qs.append(MD_QUOTE_RE.sub("", lines[i], count=1))
                i += 1
            body = strip_md_inline(join_lines(qs))
            if body:
                blocks.append((prefix + tuple(heads), "text", body + "\n"))
            continue

        # ---- ATX 标题 ----
        m = MD_ATX_RE.match(line)
        if m:
            lvl = len(m.group(1))
            t = clean_title(m.group(2))
            if t:
                heads = heads[: lvl - 1] + [t]
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        # ---- 表格：整块保留，分隔符不压平 ----
        # HTML 那边表格被 get_text 压成 "IOBase fileno , seek" 的流水，很难读；
        # md 的表格原本就是 | a | b |，原样保留反而最清楚
        if MD_TABLE_RE.match(line):
            rows: list[str] = []
            while i < n and MD_TABLE_RE.match(lines[i]):
                rows.append(lines[i].strip())
                i += 1
            blocks.append((prefix + tuple(heads), "text", "\n".join(rows) + "\n"))
            continue

        # ---- 列表：连续列表项合成一块 ----
        if MD_LIST_RE.match(line):
            items: list[str] = []
            while i < n and (MD_LIST_RE.match(lines[i]) or (lines[i].startswith("  ") and lines[i].strip())):
                items.append(strip_md_inline(lines[i]))
                i += 1
            body = "\n".join(x for x in items if x.strip())
            if body.strip():
                blocks.append((prefix + tuple(heads), "text", body + "\n"))
            continue

        # ---- 普通段落 ----
        para: list[str] = []
        while i < n:
            ln = lines[i]
            if not ln.strip():
                break
            if MD_ATX_RE.match(ln) or MD_FENCE_RE.match(ln) or MD_CONTAINER_RE.match(ln) \
                    or MD_TABLE_RE.match(ln) or MD_LIST_RE.match(ln):
                break
            para.append(ln)
            i += 1
        if para:
            txt = strip_md_inline(join_lines(para))
            if txt:
                blocks.append((prefix + tuple(heads), "text", txt + "\n"))
        else:
            i += 1     # 同样兜底，防死循环

    return blocks


# ============ 纯文本兜底 ============

def parse_text(text: str, title: str = "") -> list[Block]:
    """没有任何结构标记的纯文本：按空行分段，全部当正文。

    不猜标题、不猜代码 —— 猜错的代价比不猜大得多。
    """
    blocks: list[Block] = []
    prefix = (title,) if title else ()
    for chunk in re.split(r"\n\s*\n", text):
        body = re.sub(r"\s+", " ", chunk).strip()
        if body:
            blocks.append((prefix, "text", body + "\n"))
    return blocks


# ============ 统一入口 ============

PARSERS = {
    "rst": parse_rst,
    "md": parse_md,
    "text": parse_text,
    # html 不在表里：它要 BeautifulSoup 和站点 profile，由 parse.py 自己处理
}


def parse_markup(text: str, fmt: str, title: str = "") -> list[Block]:
    """按格式分发到对应解析器。

    返回的块必须都是 (title_path, kind, text) 三元组，
    parse.py 的打包逻辑只认这个形状 —— 这是「加格式不改主流程」的前提。
    """
    fn = PARSERS.get(fmt)
    if fn is None:
        # 未知格式当纯文本处理，绝不返回空。空块会被上层当成「这篇抽不出内容」丢掉，
        # 而实际上内容可能是好的，只是格式没认出来 —— 静默丢数据比多留噪声更糟。
        fn = parse_text
    return fn(text, title)
