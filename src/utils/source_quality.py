# -*- coding: utf-8 -*-
"""Low-quality news source filter.

Search providers sometimes return social-media posts, PDF attachments or URLs
on non-standard ports (e.g. ``benzinga.com:8443``, a Facebook post, a
governance-survey PDF served by JD Supra's file handler). They add noise to the prompt and to the
"Sources" block. ``low_quality_reason`` flags them so callers can drop them.

Extra blocked domains can be added with ``NEWS_SOURCE_BLOCKLIST`` (comma
separated, e.g. ``example.com,spam.net``); subdomains match.
"""

from __future__ import annotations

import os
from typing import FrozenSet, Optional
from urllib.parse import urlparse

DEFAULT_BLOCKED_DOMAINS: FrozenSet[str] = frozenset({
    "facebook.com",
    "fb.com",
    "instagram.com",
    "threads.net",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "linkedin.com",
    "reddit.com",
    "pinterest.com",
    "youtube.com",
})

_STANDARD_PORTS = {None, 80, 443}


def blocked_domains() -> FrozenSet[str]:
    extra = {
        part.strip().lower().lstrip(".")
        for part in os.getenv("NEWS_SOURCE_BLOCKLIST", "").split(",")
        if part.strip()
    }
    return DEFAULT_BLOCKED_DOMAINS | extra


def low_quality_reason(url: str) -> Optional[str]:
    """Return why ``url`` is a low-quality news source, or None when acceptable."""
    text = str(url or "").strip()
    if not text:
        return None
    try:
        parsed = urlparse(text)
        port = parsed.port
    except ValueError:
        return "invalid_url"
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if port not in _STANDARD_PORTS:
        return "non_standard_port"
    # Attachments served through a generic file handler (``?fName=...pdf``) are
    # usually unrelated documents; direct PDF paths are kept because exchange
    # filings (HKEX, SEC exhibits, IR releases) are often PDFs.
    if ".pdf" in parsed.query.lower():
        return "pdf_attachment"
    for domain in blocked_domains():
        if host == domain or host.endswith("." + domain):
            return "blocked_domain"
    return None
