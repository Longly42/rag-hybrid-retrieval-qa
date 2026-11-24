"""data/raw 的目录约定：格式 / 域名 / 源路径。

这一条规则只允许有一个实现。
    crawl.py 靠它决定文件落哪
    parse.py 靠它反推源 URL
    scripts/migrate_raw_layout.py 靠它搬老文件
写三遍必然漂：改了一处忘了另一处，同一篇文档就会落到两个路径，
然后你在 parse 的输出里看到「重复 URL」，开始怀疑是不是爬虫漏抓了。

约定示例：
    https://docs.python.org/3/library/bisect.html
      → html/docs.python.org/3/library/bisect.html

    https://docs.djangoproject.com/zh-hans/5.2/topics/db/models/
      → rst/docs.djangoproject.com/zh-hans/5.2/topics/db/models.txt

为什么第一层是格式而不是域名：
    同一篇文档常常有 html 和 rst 两份（渲染页 vs 源文件）。按格式分层之后，
    「这一篇抓到过几种格式」是一眼能看出来的，parse 的选优也只需比一次路径。
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# Windows 文件名里不能出现的字符。URL 路径段里偶尔会有（尤其是 : 和 ?）
BAD_SEG_RE = re.compile(r'[:*?"<>|]')
# 路径末尾已有的扩展名
ANY_EXT_RE = re.compile(r"\.(html?|txt|md|rst|text)$", re.I)
# 域名里被替换成下划线的端口号，反推时要还原
PORT_SEG_RE = re.compile(r"_(\d+)$")

# 每种格式落地时用什么扩展名。
# 注意 rst 用 .txt 而不是 .rst —— Django 的 _sources 服务器上就叫 .txt，
# 按原样存能让「文件来自哪里」这件事有据可查。
EXT_OF_FMT = {"html": ".html", "rst": ".txt", "md": ".md", "text": ".txt"}


def rel_path_of(url: str, fmt: str = "html") -> str:
    """URL → data/raw 下的相对路径（用 / 分隔，跨平台）。

    url 应该是「权威地址」：HTML 传页面里的 canonical，rst/md 传实际请求地址。
    用 canonical 而不是请求地址，是因为请求地址可能带语言前缀或查询串，
    而 canonical 是页面自己声明的身份，重爬多少次都一样。
    """
    p = urlparse(url)
    segs = [s for s in p.path.split("/") if s]
    if not segs:
        segs = ["index"]

    segs = [BAD_SEG_RE.sub("_", s) for s in segs]

    # 末尾统一扩展名：没有就补，有但和格式不符就换。
    # 换的那一步是必要的：Django 源文件在服务器上叫 .txt，内容是 rst，
    # 存成 .txt 保真；但如果有人把 .rst 文件放到 md 源里，也得按实际格式落地。
    stem = ANY_EXT_RE.sub("", segs[-1])
    segs[-1] = stem + EXT_OF_FMT.get(fmt, ".txt")

    site = p.netloc.replace(":", "_")
    return f"{fmt}/{site}/" + "/".join(segs)


def url_of_rel(rel: str) -> str:
    """相对路径 → 源 URL。这是没有 manifest 时的兜底。

    只还原得到「去掉扩展名」的地址，目录式 URL（Django 那种）会少一个结尾斜杠。
    浏览器打开会自动补，但引用显示不完美 —— 所以 crawl 要落 manifest，
    有 manifest 就用 manifest 里那条真实地址。
    """
    parts = rel.replace("\\", "/").split("/")
    if len(parts) < 3:
        return ""
    site = PORT_SEG_RE.sub(r":\1", parts[1])
    path = "/".join(parts[2:])
    path = ANY_EXT_RE.sub("", path)
    return f"https://{site}/{path}"


def format_of_rel(rel: str) -> str:
    """相对路径的第一段就是格式。"""
    return rel.replace("\\", "/").split("/")[0]


def site_of_rel(rel: str) -> str:
    """相对路径的第二段就是域名（端口已转成下划线）。"""
    parts = rel.replace("\\", "/").split("/")
    return PORT_SEG_RE.sub(r":\1", parts[1]) if len(parts) > 1 else ""
