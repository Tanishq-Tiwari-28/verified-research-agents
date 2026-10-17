"""Tools the agents call via Gemini function calling: web_search + fetch_page.
Every result is stored in an Evidence store with a stable id (S1, S2, ...) so
the Synthesizer can cite it and the Fact-Checker can verify against it."""
import json
import re

import requests
from bs4 import BeautifulSoup
from google.genai import types

try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS

UA = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgents/0.1)"}


class Evidence:
    """All sources gathered during one question, shared by every agent."""

    def __init__(self):
        self.items = []
        self._by_url = {}

    def add(self, url, title, text, date=None):
        text = (text or "").strip()
        if url in self._by_url:
            it = self._by_url[url]
            if len(text) > len(it["text"]):
                it["text"] = text
            if date and not it["date"]:
                it["date"] = date
            return it["id"]
        it = {"id": f"S{len(self.items) + 1}", "url": url, "title": title or url,
              "date": date, "text": text}
        self.items.append(it)
        self._by_url[url] = it
        return it["id"]

    def render(self, per_source=2500):
        if not self.items:
            return "(no evidence retrieved)"
        return "\n\n".join(
            f"[{i['id']}] {i['title']} | {i['url']} | published: {i['date'] or 'unknown'}\n{i['text'][:per_source]}"
            for i in self.items)

    def cited(self, answer):
        ids = set(re.findall(r"S\d+", " ".join(re.findall(r"\[([^\]]*S\d+[^\]]*)\]", answer))))
        picked = [i for i in self.items if i["id"] in ids] or self.items
        return [{k: i[k] for k in ("id", "title", "url", "date")} for i in picked]


def _find_date(soup, raw_html):
    for key in ("article:published_time", "og:article:published_time", "datePublished",
                "pubdate", "date", "DC.date.issued", "article:modified_time"):
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key}) \
            or soup.find("meta", attrs={"itemprop": key})
        if tag and tag.get("content"):
            return tag["content"][:25]
    m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', raw_html)
    if m:
        return m.group(1)[:25]
    t = soup.find("time", attrs={"datetime": True})
    return t["datetime"][:25] if t else None


S, T = types.Schema, types.Type
DECLARATIONS = [
    types.FunctionDeclaration(
        name="web_search",
        description="Search the live web. Returns titles, URLs, snippets and (for news) publication dates.",
        parameters=S(type=T.OBJECT, required=["query"], properties={
            "query": S(type=T.STRING, description="Short, specific search query."),
            "kind": S(type=T.STRING, enum=["web", "news"],
                      description="'news' returns dated recent articles; use it for anything time-sensitive."),
        })),
    types.FunctionDeclaration(
        name="fetch_page",
        description="Download a web page and return its main text plus its publication date if found.",
        parameters=S(type=T.OBJECT, required=["url"], properties={
            "url": S(type=T.STRING, description="Full http(s) URL, taken from search results."),
        })),
]


class Toolbox:
    def __init__(self, evidence: Evidence):
        self.ev = evidence
        self.declarations = DECLARATIONS

    def call(self, name, args):
        try:
            if name == "web_search":
                return self.web_search(args.get("query", ""), args.get("kind", "web"))
            if name == "fetch_page":
                return self.fetch_page(args.get("url", ""))
            return {"error": f"unknown tool {name}"}
        except Exception as e:  # tools must never crash the agent loop
            return {"error": f"{type(e).__name__}: {e}"}

    def web_search(self, query, kind="web"):
        d = DDGS()
        raw = list(d.news(query, max_results=6)) if kind == "news" else list(d.text(query, max_results=6))
        out = []
        for r in raw:
            url = r.get("href") or r.get("url")
            if not url:
                continue
            sid = self.ev.add(url, r.get("title"), r.get("body"), r.get("date"))
            out.append({"id": sid, "title": r.get("title"), "url": url,
                        "date": r.get("date"), "snippet": (r.get("body") or "")[:400]})
        return out or {"error": "no results; try a different query"}

    def fetch_page(self, url):
        if not url.startswith(("http://", "https://")):
            return {"error": "url must start with http:// or https://"}
        r = requests.get(url, headers=UA, timeout=12)
        if "html" not in r.headers.get("content-type", ""):
            return {"error": "not an HTML page"}
        soup = BeautifulSoup(r.text, "html.parser")
        date = _find_date(soup, r.text)
        title = soup.title.get_text(strip=True) if soup.title else url
        for t in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            t.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
        if not text:
            return {"error": "page had no readable text"}
        sid = self.ev.add(url, title, text[:6000], date)
        return {"id": sid, "title": title, "date": date, "text": text[:3500]}
