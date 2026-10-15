"""PMC OA 全文抓取：esearch 拿 ID → efetch 拿全文 XML → 落盘。

只负责下载，不做解析（解析在 parse_pmc.py）。

三处设计要点，都是"要抓 1000 篇"这件事逼出来的 —— 面试问
「你怎么保证一个跑几十分钟的任务不中断」时，答的就是这三条：

1. **限速间隔从配置推导，不写死。**
   NCBI 规定匿名 3 req/s、带 API key 10 req/s，间隔取 `config.NCBI_MIN_INTERVAL`。
   写死 0.4 的代价：哪天你申请了 key，代码还按 3 req/s 爬，白等；
   哪天规则收紧，写死的值就从"安全"变成"限流器"。

2. **断点续传按"文件名是否已存在"判断。**
   跑到第 40 分钟断网，重启后应该只补没抓到的部分。
   注意判断用的是 **esearch 给的数字 id**，它和 efetch 返回的 pmcaid 一致 ——
   这也是文件名不用 `<article id="...">` 的原因（那是出版社内部稿号，换期刊就换格式）。

3. **失败隔离：单批异常只记账，不 raise。**
   长任务里必然有抖动。让一次抖动毁掉前面半小时是不可接受的。
   失败的 id 落盘 `_failed.txt`，而且**下次重跑会自动重试它们**（因为文件不存在）。

实测记录（2026-10-15）：
  - `esearch(db=pmc)` 返回纯数字 ID，不带 PMC 前缀
  - efetch 的根元素**恒为 `<pmc-articleset>`**，取单个 id 也是 —— 里面套 1 个 `<article>`
  - 非 OA 全文的 id 会返回一个空 articleset（0 个 `<article>`），**不报错**
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import requests
from lxml import etree

from src.config import (
    NCBI_API_KEY,
    NCBI_EMAIL,
    NCBI_MIN_INTERVAL,
    RAW_DIR,
    ensure_dirs,
    setup_logging,
)

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH_SIZE = 200                 # efetch 每批上限（周任务卡定死）
OA_FILTER = '"open access"[filter]'
FAILED_LIST_NAME = "_failed.txt"
API_KEY_INVALID_MSG = "API key invalid"

# 取 pmcaid 的两条候选路径：优先数字形式，退回带 PMC 前缀的形式
_ID_XPATHS = (
    ('./front/article-meta/article-id[@pub-id-type="pmcaid"]', ""),
    ('./front/article-meta/article-id[@pub-id-type="pmcid"]', "PMC"),
)


def _params(extra: dict) -> dict:
    """E-utilities 公共参数。

    tool / email 是官方要求的"礼貌参数"：不填也能跑，但被限流时没有申诉依据。
    ⚠️ 一律用 requests 的 params= 传参，**别手拼 URL 字符串** ——
    '"open access"[filter]' 里的引号和方括号手拼必出编码问题。
    """
    params = {"db": "pmc", "tool": "medevidence", "email": NCBI_EMAIL}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    params.update(extra)
    return params


def esearch_pmc(
    term: str,
    retmax: int = 1000,
    *,
    sort: str | None = None,
    mindate: str | None = None,
    session: requests.Session | None = None,
) -> list[str]:
    """检索 PMC，返回纯数字 ID 列表。

    term 会自动拼上 OA 过滤（技术方案 §2 定死）——
    不加它，返回的文献里有相当一部分拿不到全文，白抓。

    sort 可选 'pub_date'（最新）或 'relevance'（最相关）。
    ⚠️ 不传时 PMC 的默认排序**就是 pub_date**（2026-10-15 实测：sort=None 与
    sort=pub_date 返回的 id 列表完全相同）。所以"默认即最新"——
    这意味着每次查询拿到的都是一个**会移动的集合**，语料必须另外冻结（见 plan/04）。

    这里**不做翻页**：retmax 上限 10000，本项目的量级用不到。
    等真需要更多再写（第 5 周扩语料时大概率会遇到）。
    """
    params = _params({
        "term": f"{term} AND {OA_FILTER}",
        "retmax": retmax,
        "retmode": "json",
    })
    if sort:
        params["sort"] = sort
    if mindate:
        params["mindate"] = mindate
        params["datetype"] = "pdat"
    try:
        resp = (session or requests.Session()).get(
            f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30
        )
        resp.raise_for_status()
        ids = resp.json()["esearchresult"]["idlist"]
        logger.info("esearch「%s」命中 %d 篇", term, len(ids))
        return ids
    except requests.HTTPError as e:
        body = e.response.text[:500] if e.response is not None else "无响应"

        if API_KEY_INVALID_MSG in body:
            logger.info("你的NCBI_API_KEY 无效，重试匿名帐户")
            logger.info("等待 %f 秒后重试", NCBI_MIN_INTERVAL)
            time.sleep(NCBI_MIN_INTERVAL) # 避免被限流，退避
            #重试，换回匿名帐户
            params.pop("api_key", None)
            resp = (session or requests.Session()).get(
                f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30
            )
            resp.raise_for_status()
            ids = resp.json()["esearchresult"]["idlist"]
            logger.info("esearch（匿名）「%s」命中 %d 篇", term, len(ids))
            return ids
        else:   
            logger.error(
                "esearch HTTP 发生未知错误：%s；term=%r；body=%s",
                e, term, body,
            )
            raise

    except Exception as e:
        logger.error("esearch「%s」失败：%s", term, e)
        raise


def _pmcaid(article) -> str | None:
    """从一篇文章元素里取它自己的编号（数字形式）。"""
    for xpath, prefix in _ID_XPATHS:
        node = article.find(xpath)
        if node is not None and node.text and node.text.strip():
            value = node.text.strip()
            if prefix and value.startswith(prefix):
                return value[len(prefix):]
            return value
    return None


def efetch_batch(
    ids: list[str],
    *,
    session: requests.Session | None = None,
) -> list[tuple[str, bytes]]:
    """取一批全文，返回 [(pmcaid, 单篇 `<article>` 元素的字节), ...]。

    ⚠️ efetch 的根元素恒为 `<pmc-articleset>`（**单个 id 也是**），所以必须
    `findall(".//article")`，**不能拿 root 当 article** —— 那样取不到任何字段，
    而且不报错，会静默产出一堆空文章。这是最坏的一类 bug：不崩、数据全空。

    返回空列表（0 篇）是**合法结果**：非 OA 全文的 id 就是这样，
    交给上层记账，不当异常处理。
    """
    params = _params({"id": ",".join(ids), "retmode": "xml"})
    resp = (session or requests.Session()).get(
        f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=120
    )
    resp.raise_for_status()

    root = etree.fromstring(resp.content)
    out: list[tuple[str, bytes]] = []
    for article in root.findall(".//article"):
        pmcaid = _pmcaid(article)
        if pmcaid is None:
            logger.warning("跳过一篇文章：既没有 pmcaid 也没有 pmcid")
            continue
        out.append((pmcaid, etree.tostring(article)))
    return out


def fetch_to_raw(
    ids: list[str],
    out_dir: Path = RAW_DIR,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> dict:
    """主循环：分批下载到 out_dir/{数字id}.xml，返回 {'ok', 'skip', 'fail'} 统计。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()

    stats = {"ok": 0, "skip": 0, "fail": 0}
    failed: list[str] = []
    batches = [ids[i : i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]

    for n, batch in enumerate(batches, 1):
        # 断点续传：已落盘的跳过。force=True 时全部重下
        todo = list(batch) if force else [
            i for i in batch if not (out_dir / f"{i}.xml").exists()
        ]
        stats["skip"] += len(batch) - len(todo)
        if not todo:
            continue

        logger.info("批次 %d/%d：需下载 %d 篇", n, len(batches), len(todo))
        try:
            pairs = efetch_batch(todo, session=session)
        except Exception as err:  # noqa: BLE001 —— 见模块头第 3 条
            logger.error("批次 %d/%d 整批失败（%d 篇）：%s", n, len(batches), len(todo), err)
            pairs = None

        if pairs is None:
            stats["fail"] += len(todo)
            failed.extend(todo)          # 文件没落盘 → 下次重跑会自动重试它们
        else:
            saved = set()
            for pmcaid, xml in pairs:
                (out_dir / f"{pmcaid}.xml").write_bytes(xml)
                saved.add(pmcaid)
                stats["ok"] += 1
            missing = [i for i in todo if i not in saved]
            if missing:
                logger.warning("批次 %d/%d 有 %d 篇没返回全文：%s",
                               n, len(batches), len(missing), missing)
                stats["fail"] += len(missing)
                failed.extend(missing)

        # 限速：批与批之间等一会儿。值来自 config（有 key 就是 10 req/s）
        if n < len(batches):
            time.sleep(NCBI_MIN_INTERVAL)

    _write_failed_list(out_dir, failed)
    logger.info("抓取结束：新增 %d / 跳过 %d / 失败 %d",
                stats["ok"], stats["skip"], stats["fail"])
    return stats


def _write_failed_list(out_dir: Path, failed: list[str]) -> None:
    """失败清单落盘；本轮没失败就删掉旧的。

    留着上一轮的清单不是"多点垃圾"——第 7 周做 badcase 归因时，
    你会把上一轮失败的 id 当成这一轮的，整张统计表都是错的。
    """
    path = out_dir / FAILED_LIST_NAME
    if failed:
        path.write_text("\n".join(failed) + "\n", encoding="utf-8")
    elif path.exists():
        path.unlink()


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="抓取 PMC OA 全文到 data/raw/")
    parser.add_argument("--term", default='"Image Processing, Computer-Assisted"[MeSH Terms] AND "Pathology"[MeSH Terms]', help="检索词（自动加 OA 过滤）")
    parser.add_argument("--limit", type=int, default=10, help="抓多少篇")
    parser.add_argument("--force", action="store_true", help="已存在的也重新下载")
    args = parser.parse_args(argv)

    ensure_dirs()
    ids = esearch_pmc(args.term, retmax=args.limit,sort="relevance")
    if not ids:
        print("esearch 没有命中任何文献")
        return

    stats = fetch_to_raw(ids, force=args.force)
    print(f"完成：新增 {stats['ok']} / 跳过 {stats['skip']} / 失败 {stats['fail']}")


if __name__ == "__main__":
    main()
