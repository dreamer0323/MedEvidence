"""抓取层测试 —— 全部离线，用假 session 顶替 HTTP，不碰网络。

为什么这层值得测：抓取的 bug 全是"静默"的 —— 少拆出一篇、断点续传失效导致
重复下载、单批失败把整个任务带崩。这三类都不会在本地跑 10 篇时暴露，
等跑到 1000 篇、跑到第 40 分钟才炸，那时你已经不知道是哪一批的问题了。
"""
from __future__ import annotations

import pytest
import requests

from src.ingest import fetch_pmc

# ------------------------------------------------------------ 假的 HTTP 层


def articleset(*ids: str) -> bytes:
    """按给定的 pmcaid 造一个 pmc-articleset 响应体（模仿 efetch 的真实形状）。"""
    body = "".join(
        f"""<article>
      <front><article-meta>
        <article-id pub-id-type="pmcaid">{i}</article-id>
        <article-id pub-id-type="pmcid">PMC{i}</article-id>
      </article-meta></front>
      <body><sec><title>Methods</title><p>text-{i}</p></sec></body>
    </article>"""
        for i in ids
    )
    return f'<?xml version="1.0" ?><pmc-articleset>{body}</pmc-articleset>'.encode()


# 真实观测：非 OA 全文的 id 会返回一个空 articleset（0 个 <article>）
EMPTY_ARTICLESET = b'<?xml version="1.0" ?><pmc-articleset></pmc-articleset>'


class FakeResponse:
    def __init__(self, content: bytes = b"", status: int = 200, payload: dict | None = None):
        self.content = content
        self.status_code = status
        self._payload = payload

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self          # requests 的真实行为：异常上挂着 response
            raise err

    def json(self):
        return self._payload


def sent_url(call) -> str:
    """把 (url, params) 按 requests 的真实编码规则还原成最终发出的那条 URL。

    为什么不用 `params` 字典断言：`params={"a": None}` 在 requests 里**会被丢掉**，
    dict 上看着还在、线上其实没发。断真实 URL 才是断行为。
    """
    url, params = call
    return requests.Request("GET", url, params=params).prepare().url


class FakeSession:
    """按顺序吐出预置响应；同时记下每次请求的参数，供断言观察 HTTP 边界。"""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item



# ------------------------------------------------------------ 切片 1：esearch


def test_esearch_returns_idlist_and_applies_oa_filter():
    session = FakeSession([
        FakeResponse(payload={"esearchresult": {"idlist": ["1", "2", "3"]}})
    ])

    ids = fetch_pmc.esearch_pmc("pathology image", session=session)

    assert ids == ["1", "2", "3"]
    url, params = session.calls[0]
    assert url.endswith("/esearch.fcgi")
    assert params["db"] == "pmc"
    assert params["retmode"] == "json"
    # 技术方案 §2 定死：必须带 OA 过滤，否则会抓到拿不到全文的文章
    assert '"open access"[filter]' in params["term"]


# ------------------------------------------------------------ 切片 2：efetch 拆包


def test_efetch_batch_splits_articleset_into_articles():
    session = FakeSession([FakeResponse(content=articleset("13660408", "13659917"))])

    got = fetch_pmc.efetch_batch(["13660408", "13659917"], session=session)

    assert [pmcaid for pmcaid, _ in got] == ["13660408", "13659917"]
    # 拆出来的必须是**单篇** <article> 的字节，不是整个 articleset
    assert b"<pmc-articleset>" not in got[0][1]
    assert b"text-13660408" in got[0][1]
    assert b"13659917" not in got[0][1]


def test_efetch_batch_returns_empty_on_empty_articleset():
    """0 篇是合法响应（非 OA / id 无效），不是异常 —— 上层要靠它记失败。"""
    session = FakeSession([FakeResponse(content=EMPTY_ARTICLESET)])

    assert fetch_pmc.efetch_batch(["999"], session=session) == []


# ------------------------------------------------------------ 切片 3：主循环


def test_fetch_to_raw_skips_already_downloaded(tmp_path):
    """断点续传：文件已在 → 一个请求都不该发。"""
    (tmp_path / "13660408.xml").write_bytes(b"<already/>")
    session = FakeSession([])          # 没有预置响应；一旦发请求就会 IndexError

    stats = fetch_pmc.fetch_to_raw(["13660408"], out_dir=tmp_path, session=session)

    assert stats == {"ok": 0, "skip": 1, "fail": 0}
    assert session.calls == []


def test_fetch_to_raw_writes_file_named_by_pmcaid(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_pmc, "NCBI_MIN_INTERVAL", 0)   # 测试里不等
    session = FakeSession([FakeResponse(content=articleset("13660408"))])

    stats = fetch_pmc.fetch_to_raw(["13660408"], out_dir=tmp_path, session=session)

    assert stats == {"ok": 1, "skip": 0, "fail": 0}
    assert (tmp_path / "13660408.xml").exists()


def test_fetch_to_raw_isolates_failed_batch(tmp_path, monkeypatch):
    """一批挂了不能带崩整个任务，且失败 id 要落盘 —— 后面 badcase 归因要用。"""
    monkeypatch.setattr(fetch_pmc, "BATCH_SIZE", 2)
    monkeypatch.setattr(fetch_pmc, "NCBI_MIN_INTERVAL", 0)
    session = FakeSession([
        requests.ConnectionError("boom"),                       # 第 1 批：整批失败
        FakeResponse(content=articleset("13659184", "13659099")),  # 第 2 批：正常
    ])

    stats = fetch_pmc.fetch_to_raw(
        ["13660408", "13659917", "13659184", "13659099"], out_dir=tmp_path, session=session
    )

    assert stats["fail"] == 2                       # 第 1 批全失败
    assert stats["ok"] == 2                         # 第 2 批照常落盘
    assert (tmp_path / "13659184.xml").exists()
    failed = (tmp_path / "_failed.txt").read_text(encoding="utf-8").split()
    assert failed == ["13660408", "13659917"]


def test_fetch_to_raw_clears_stale_failed_list(tmp_path, monkeypatch):
    """上一轮留下的失败清单必须清掉。

    不清的后果不是"多点垃圾" —— 是第 7 周做 badcase 归因时，你会把上一轮的
    失败 id 当成这一轮的失败，整张统计表都是错的。
    """
    (tmp_path / "_failed.txt").write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(fetch_pmc, "NCBI_MIN_INTERVAL", 0)
    session = FakeSession([FakeResponse(content=articleset("13660408"))])

    fetch_pmc.fetch_to_raw(["13660408"], out_dir=tmp_path, session=session)

    assert not (tmp_path / "_failed.txt").exists()


# ------------------------------------------------------------ 切片 4：重试与降级

# 2026-10-15 实测抓到的真实响应体（NCBI 在拒服务时就是这个形状）
BAD_KEY_BODY = b'{"error":"API key invalid","api-key":"bad-key","type":"invalid"}'


def test_esearch_retries_anonymously_when_api_key_is_rejected(monkeypatch):
    """带了一个坏 key 时，应该退回匿名再试一次，而不是整个任务挂掉。"""
    monkeypatch.setattr(fetch_pmc, "NCBI_API_KEY", "bad-key")
    monkeypatch.setattr(fetch_pmc.time, "sleep", lambda _s: None)
    session = FakeSession([
        FakeResponse(status=400, content=BAD_KEY_BODY),
        FakeResponse(payload={"esearchresult": {"idlist": ["1", "2"]}}),
    ])

    ids = fetch_pmc.esearch_pmc("pathology", session=session)

    assert ids == ["1", "2"]
    assert "api_key=bad-key" in sent_url(session.calls[0])     # 第一次带了 key
    assert "api_key" not in sent_url(session.calls[1])         # 第二次是匿名


def test_esearch_does_not_blame_the_api_key_for_other_errors(monkeypatch):
    """500 跟 key 没关系，不该被当成 key 问题去重试匿名。

    这是我们今天撞过的坑：NCBI 拒服务时回的是 "API key invalid"，
    而请求里根本没有 key。**报错信息会撒谎**，所以不能见到 4xx/5xx 就归因到 key。
    """
    monkeypatch.setattr(fetch_pmc, "NCBI_API_KEY", "bad-key")
    session = FakeSession([FakeResponse(status=500, content=b"upstream exploded")])

    with pytest.raises(requests.HTTPError):
        fetch_pmc.esearch_pmc("pathology", session=session)

    assert len(session.calls) == 1        # 没有第二次请求


def test_esearch_backs_off_before_retrying(monkeypatch):
    """重试之前必须等一会儿。

    失败后立刻重打，是把"被限流"升级成"被封 IP"的经典做法 ——
    第 11 周的降级链路要做的也是这件事。
    """
    monkeypatch.setattr(fetch_pmc, "NCBI_API_KEY", "bad-key")
    slept: list[float] = []
    monkeypatch.setattr(fetch_pmc.time, "sleep", slept.append)
    session = FakeSession([
        FakeResponse(status=400, content=BAD_KEY_BODY),
        FakeResponse(payload={"esearchresult": {"idlist": []}}),
    ])

    fetch_pmc.esearch_pmc("pathology", session=session)

    assert slept and slept[0] > 0


def test_esearch_passes_sort_through():
    """sort 参数 —— 「排序到底影响什么」那个实验要用它（这一处是我欠的）。"""
    session = FakeSession([FakeResponse(payload={"esearchresult": {"idlist": []}})])

    fetch_pmc.esearch_pmc("pathology", session=session, sort="relevance")

    assert session.calls[0][1]["sort"] == "relevance"
