"""
Scraper — fetches Anthropic Claude API documentation pages and converts to clean markdown.

Why: architecture.md §2.1 requires fetching, cleaning, and caching doc pages.
Respects robots.txt, caches raw HTML locally (rules.md §2), extracts structured
React RSC / Markdoc AST streams (with fallback to BeautifulSoup HTML parsing),
preserving headings and code blocks for structure-aware chunking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from src.config import RAW_DATA_DIR, TARGET_DOC_URLS

logger = logging.getLogger(__name__)

_ROBOT_PARSER_CACHE: dict[str, RobotFileParser] = {}


def _get_robot_parser(url: str) -> RobotFileParser:
    """Fetch and cache robots.txt for a given URL's domain."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _ROBOT_PARSER_CACHE:
        rp = RobotFileParser()
        try:
            resp = requests.get(f"{base}/robots.txt", timeout=10)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                rp.set_url(f"{base}/robots.txt")
                rp.read()
        except Exception:
            logger.warning("Could not fetch robots.txt for %s, allowing doc paths", base)
            rp.allow_all = True
        _ROBOT_PARSER_CACHE[base] = rp
    return _ROBOT_PARSER_CACHE[base]


def _can_fetch(url: str) -> bool:
    """Check robots.txt permission for the given URL."""
    try:
        rp = _get_robot_parser(url)
        return rp.can_fetch("*", url)
    except Exception:
        return True


def _url_to_cache_path(url: str, suffix: str = ".html") -> Path:
    """Deterministic local cache path for a URL's raw source."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
    return RAW_DATA_DIR / f"{slug}_{url_hash}{suffix}"


def _markdown_mirror_url(url: str) -> str | None:
    """Return Anthropic's official markdown mirror URL for a docs page."""
    parsed = urlparse(url)
    if not parsed.path.startswith(("/docs/en/", "/en/")) or parsed.path.endswith(".md"):
        return None
    return urlunparse(parsed._replace(path=f"{parsed.path}.md", query="", fragment=""))


def fetch_page(url: str, use_cache: bool = True) -> str | None:
    """
    Fetch a single documentation page. Returns raw HTML or None on failure.
    Caches to data/raw/ to avoid repeat fetches (rules.md §2).
    """
    html_cache_path = _url_to_cache_path(url, ".html")
    markdown_cache_path = _url_to_cache_path(url, ".md")

    if use_cache:
        for cache_path in (markdown_cache_path, html_cache_path):
            if cache_path.exists():
                logger.debug("Cache hit: %s", url)
                return cache_path.read_text(encoding="utf-8")

    if not _can_fetch(url):
        logger.warning("Blocked by robots.txt: %s", url)
        return None

    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to fetch %s: %s", url, exc)
        return None

    mirror_url = _markdown_mirror_url(resp.url)
    if mirror_url:
        try:
            mirror_resp = requests.get(
                mirror_url,
                headers={
                    "User-Agent": "Colophon educational docs indexer",
                    "Accept": "text/markdown,text/plain;q=0.9,*/*;q=0.8",
                },
                timeout=30,
            )
            if (
                mirror_resp.status_code == 200
                and "markdown" in mirror_resp.headers.get("Content-Type", "")
                and len(mirror_resp.text.strip()) > 100
            ):
                markdown_cache_path.write_text(mirror_resp.text, encoding="utf-8")
                logger.info("Fetched and cached markdown mirror: %s", url)
                return mirror_resp.text
        except requests.RequestException as exc:
            logger.debug("Markdown mirror unavailable for %s: %s", url, exc)

    html = resp.text
    html_cache_path.write_text(html, encoding="utf-8")
    logger.info("Fetched and cached: %s", url)
    return html


def _render_rsc_node(elem: Any) -> str:
    """Recursively convert Next.js React Server Component element tree to Markdown."""
    if isinstance(elem, str):
        if elem.startswith("$") and len(elem) < 15:
            return ""
        return elem
    if isinstance(elem, (int, float)):
        return str(elem)
    if isinstance(elem, list):
        if len(elem) >= 4 and elem[0] == "$":
            tag = elem[1]
            props = elem[3] if isinstance(elem[3], dict) else {}
            children = props.get("children", [])
            
            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(tag[1])
                t = _render_rsc_node(children).strip()
                return f"\n\n{'#' * level} {t}\n\n" if t else ""
            elif tag == "p":
                t = _render_rsc_node(children).strip()
                return f"\n\n{t}\n\n" if t else ""
            elif tag == "code":
                t = _render_rsc_node(children)
                return f"`{t}`" if t else ""
            elif tag in ("pre", "CodeBlock", "CodeSnippet"):
                t = _render_rsc_node(children)
                return f"\n\n```\n{t.strip()}\n```\n\n" if t else ""
            elif tag == "li":
                return _render_rsc_node(children)
            elif tag in ("ul", "ol"):
                items = []
                if isinstance(children, list):
                    for c in children:
                        t = _render_rsc_node(c).strip()
                        if t:
                            items.append(f"- {t}")
                return "\n" + "\n".join(items) + "\n"
            elif tag == "a":
                href = props.get("href", "")
                t = _render_rsc_node(children).strip()
                return f"[{t}]({href})" if t and href else t
            else:
                return _render_rsc_node(children)
        else:
            return "".join(_render_rsc_node(x) for x in elem)
    if isinstance(elem, dict):
        if "children" in elem:
            return _render_rsc_node(elem["children"])
        return ""
    return ""


def _extract_from_next_rsc(html: str) -> str | None:
    """Extract and parse markdown from Next.js self.__next_f App Router payloads."""
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script")
    
    for s in scripts:
        txt = s.string or ""
        if "self.__next_f.push" in txt and len(txt) > 10000:
            idx = txt.find("push([1,")
            if idx != -1:
                start = idx + len("push([1,")
                try:
                    raw_str, _ = json.JSONDecoder().raw_decode(txt[start:])
                    parts = []
                    for line in raw_str.split("\n"):
                        colon = line.find(":")
                        if colon != -1:
                            body = line[colon + 1 :]
                            if body.startswith("["):
                                try:
                                    data = json.loads(body)
                                    rendered = _render_rsc_node(data).strip()
                                    if len(rendered) > 15:
                                        parts.append(rendered)
                                except Exception:
                                    pass
                    if parts:
                        return "\n\n".join(parts)
                except Exception as exc:
                    logger.debug("Failed to decode Next.js RSC: %s", exc)
    return None


def _extract_from_initial_state(html: str) -> str | None:
    """Extract and render markdown from window.__INITIAL_STATE__ if present."""
    marker = "window.__INITIAL_STATE__ = "
    idx = html.find(marker)
    if idx == -1:
        marker = "window.__INITIAL_STATE__="
        idx = html.find(marker)
        if idx == -1:
            return None

    try:
        start_pos = idx + len(marker)
        decoder = json.JSONDecoder()
        state, _ = decoder.raw_decode(html[start_pos:])
        article = state.get("article", {})
        content = article.get("content") or article.get("body")
        if not content:
            return None

        title = article.get("metadata", {}).get("title") or article.get("title", "")
        return f"# {title}\n\n{str(content)}"
    except Exception:
        return None


def _extract_main_content(soup: BeautifulSoup) -> Tag | None:
    """Find the main content area, stripping nav/footer/sidebar."""
    for selector in ["article", "main", '[role="main"]', "#main-content", ".content", "#content"]:
        content = soup.select_one(selector)
        if content:
            return content
    return soup.body


def _tag_to_markdown(tag: Tag | NavigableString, depth: int = 0) -> str:
    """Recursively convert an HTML tag tree to clean markdown."""
    if isinstance(tag, NavigableString):
        return re.sub(r"\s+", " ", str(tag))

    name = tag.name
    if name in ("nav", "footer", "header", "script", "style", "svg", "aside"):
        return ""
    if tag.get("aria-hidden") == "true":
        return ""

    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        text = tag.get_text(strip=True)
        return f"\n\n{'#' * level} {text}\n\n" if text else ""

    if name == "pre":
        code_tag = tag.find("code")
        code_text = code_tag.get_text() if code_tag else tag.get_text()
        lang = ""
        if code_tag and code_tag.get("class"):
            for cls in code_tag["class"]:
                if cls.startswith("language-"):
                    lang = cls.replace("language-", "")
                    break
        return f"\n\n```{lang}\n{code_text.strip()}\n```\n\n"

    if name == "code" and tag.parent and tag.parent.name != "pre":
        return f"`{tag.get_text()}`"

    if name in ("ul", "ol"):
        items = []
        for i, li in enumerate(tag.find_all("li", recursive=False)):
            prefix = f"{i + 1}." if name == "ol" else "-"
            text = _tag_to_markdown(li, depth + 1).strip()
            items.append(f"{prefix} {text}")
        return "\n" + "\n".join(items) + "\n"

    if name == "table":
        rows = tag.find_all("tr")
        if not rows:
            return ""
        md_rows: list[str] = []
        for row in rows:
            cells = row.find_all(["th", "td"])
            md_rows.append("| " + " | ".join(c.get_text(strip=True) for c in cells) + " |")
            if row.find("th") and len(md_rows) == 1:
                md_rows.append("| " + " | ".join("---" for _ in cells) + " |")
        return "\n" + "\n".join(md_rows) + "\n"

    if name == "p":
        text = "".join(_tag_to_markdown(c, depth) for c in tag.children).strip()
        return f"\n\n{text}\n\n" if text else ""

    if name == "a":
        text = tag.get_text(strip=True)
        href = tag.get("href", "")
        return f"[{text}]({href})" if text and href else text

    if name in ("strong", "b"):
        return f"**{tag.get_text(strip=True)}**"
    if name in ("em", "i"):
        return f"*{tag.get_text(strip=True)}*"

    return "".join(_tag_to_markdown(c, depth) for c in tag.children)


def html_to_markdown(html: str, source_url: str) -> str:
    """
    Convert raw HTML to clean markdown.
    Attempts Next.js RSC extraction first, then INITIAL_STATE, then BeautifulSoup DOM.
    """
    stripped = html.strip()
    if (
        (stripped.startswith("#") or stripped.startswith("---\n"))
        and "<!DOCTYPE html" not in stripped[:200].lower()
    ):
        markdown = stripped
        if markdown.startswith("---\n"):
            match = re.match(r"^---\n(?P<frontmatter>.*?)\n---\n*", markdown, re.DOTALL)
            if match:
                frontmatter = match.group("frontmatter")
                title_match = re.search(r"(?m)^title:\s*(.+)$", frontmatter)
                title = title_match.group(1).strip().strip('"') if title_match else ""
                markdown = markdown[match.end() :].strip()
                if title and not markdown.startswith("#"):
                    markdown = f"# {title}\n\n{markdown}"
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
        return f"<!-- source: {source_url} -->\n\n{markdown}"

    markdown = _extract_from_next_rsc(html)

    if not markdown or len(markdown.strip()) < 100:
        markdown = _extract_from_initial_state(html)

    if not markdown or len(markdown.strip()) < 100:
        soup = BeautifulSoup(html, "html.parser")
        main = _extract_main_content(soup)
        if main is not None:
            markdown = _tag_to_markdown(main)
        else:
            markdown = ""

    # Normalize whitespace
    markdown = re.sub(r"(?m)^\$[A-Za-z0-9]+(?::[A-Za-z0-9]+)+.*\n?", "", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = markdown.strip()

    return f"<!-- source: {source_url} -->\n\n{markdown}"


def _detect_doc_category(url: str) -> str:
    """Derive a coarse category from the Claude doc URL."""
    path = urlparse(url).path.lower()
    if "tool" in path or "agent" in path:
        return "tools_and_agents"
    if "thinking" in path:
        return "extended_thinking"
    if "message" in path or "streaming" in path or "batch" in path:
        return "messages_api"
    if "prompt" in path:
        return "prompt_engineering"
    if "model" in path or "pricing" in path:
        return "models_and_pricing"
    if "mcp" in path:
        return "mcp"
    if "vision" in path or "files" in path:
        return "multimodal"
    if "rate" in path or "auth" in path or "error" in path or "manage" in path:
        return "management"
    return "general"


def scrape_all(
    urls: list[str] | None = None, use_cache: bool = True
) -> list[dict[str, str]]:
    """
    Scrape all target documentation pages and return cleaned markdown documents.
    Returns a list of dicts: {"url", "markdown", "category"}
    """
    urls = urls or TARGET_DOC_URLS
    documents: list[dict[str, str]] = []

    for url in urls:
        html = fetch_page(url, use_cache=use_cache)
        if html is None:
            continue

        markdown = html_to_markdown(html, url)
        if not markdown.strip() or len(markdown.strip()) < 50:
            logger.warning("Empty or insufficient content after cleaning: %s", url)
            continue

        documents.append(
            {
                "url": url,
                "markdown": markdown,
                "category": _detect_doc_category(url),
            }
        )
        time.sleep(0.2)

    logger.info("Scraped %d documents from %d URLs", len(documents), len(urls))
    return documents
