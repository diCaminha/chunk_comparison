# updated_llm_chunker.py  ·  23 Jul 2025
# ---------------------------------------------------------------
#  Resilient LLM‑based semantic chunker for LangChain Documents
# ---------------------------------------------------------------

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tiktoken
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ------------------------------------------------------------------
# Load environment variables **before** creating the OpenAI client
# ------------------------------------------------------------------
load_dotenv()  # expects OPENAI_API_KEY in .env or already in the shell

# ------------------------------------------------------------------
# LangChain Document import (compatible across LC versions)
# ------------------------------------------------------------------
try:
    from langchain_core.documents import Document
except ImportError:  # <0.1.0
    from langchain.schema import Document  # type: ignore

from langchain_openai import ChatOpenAI

# ================================================================
# Configuration
# ================================================================
@dataclass
class LlmChunkerConfig:
    target_tokens: int = 700
    max_tokens: int = 1000
    hard_cap_tokens: int = 1200
    enc_name: str = "cl100k_base"
    window_chars: int = 15_000
    overlap_chars: int = 800
    include_titles: bool = True
    method_tag: str = "semantic_llm"
    request_timeout: int = 60        # seconds per OpenAI request
    max_retries: int = 3             # network retries (handled by tenacity)


# ================================================================
# Token helpers
# ================================================================
def _get_encoder(name: str):
    return tiktoken.get_encoding(name)


def _count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


# ================================================================
# Window helper  (slide a large doc through the LLM)
# ================================================================
def _window_text(text: str, win: int, overlap: int) -> List[Tuple[int, str]]:
    out, n, start = [], len(text), 0
    while start < n:
        end = min(n, start + win)
        out.append((start, text[start:end]))
        if end == n:
            break
        start = max(0, end - overlap)
    return out


# ================================================================
# Prompt construction
# ================================================================
_PROMPT_TEMPLATE = """Segment the following TEXT into topically coherent sections for retrieval.

Return **ONLY** valid JSON of the form:
{{
  "segments": [
    {{"start_char": <int>, "end_char": <int>, "title": "<short title>"}},
    ...
  ]
}}

Rules:
- Offsets are 0‑based character indexes **within the TEXT block below**.
- Cover the entire text; no gaps; no overlaps; maintain order.
- Aim for ~{target_tokens} tokens per segment (soft); never exceed {max_tokens} tokens;
  split again if a segment would exceed {hard_cap_tokens}.
- Prefer cutting at sentence boundaries; avoid mid‑sentence cuts unless necessary.
- Titles: ≤ 10 words; empty string if uncertain.

TEXT:
\"\"\"{text}\"\"\""""


def _build_messages(text_block: str, cfg: LlmChunkerConfig) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert in information retrieval. "
                "You segment long text into coherent sections to optimise retrieval‑augmented generation."
            ),
        },
        {
            "role": "user",
            "content": _PROMPT_TEMPLATE.format(
                target_tokens=cfg.target_tokens,
                max_tokens=cfg.max_tokens,
                hard_cap_tokens=cfg.hard_cap_tokens,
                text=text_block,
            ),
        },
    ]


# ================================================================
# Robust JSON extraction & parsing
# ================================================================
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str, full_len: int) -> str:
    """Return first valid JSON blob; fallback to a full‑span stub."""
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if m:
        blob = m.group(0)
        try:
            json.loads(blob)
            return blob
        except Exception:
            pass
    return f'{{"segments":[{{"start_char":0,"end_char":{full_len},"title":"FullSpan"}}]}}'


def _parse_segments(raw_text: str, text_len: int) -> List[Tuple[int, int, str]]:
    segs_raw = json.loads(_extract_json(raw_text, text_len)).get("segments", [])
    segs: List[Tuple[int, int, str]] = []
    for seg in segs_raw:
        try:
            s = max(0, int(seg["start_char"]))
            e = min(text_len, int(seg["end_char"]))
            if e > s:
                segs.append((s, e, str(seg.get("title", "")).strip()))
        except Exception:
            continue
    if not segs:
        segs = [(0, text_len, "FullSpan")]

    # sort & deduplicate & fill gaps
    segs.sort(key=lambda x: x[0])
    cleaned, last_end = [], 0
    for s, e, t in segs:
        if s > last_end:
            cleaned.append((last_end, s, "AutoFill"))
        s = max(s, last_end)
        cleaned.append((s, e, t))
        last_end = e
    if last_end < text_len:
        cleaned.append((last_end, text_len, "AutoTail"))

    # merge adjacent Auto* slices
    merged: List[Tuple[int, int, str]] = []
    for s, e, t in cleaned:
        if merged and t.startswith("Auto") and merged[-1][2].startswith("Auto"):
            ps, pe, pt = merged[-1]
            merged[-1] = (ps, e, pt)
        else:
            merged.append((s, e, t))
    return merged


# ================================================================
# Linear‑time fallback splitter (token aware)
# ================================================================
def _split_segment_tokenwise(text: str, enc, max_tok: int) -> List[Tuple[int, int]]:
    tokens = enc.encode(text)
    if len(tokens) <= max_tok:
        return [(0, len(text))]
    spans, char_start, i, n = [], 0, 0, len(tokens)
    while i < n:
        j = min(n, i + max_tok)
        chunk = enc.decode(tokens[i:j])
        char_end = char_start + len(chunk)
        spans.append((char_start, char_end))
        char_start, i = char_end, j
    return spans


# ================================================================
# Merge overlapping spans coming from multiple windows
# ================================================================
def _merge_overlaps(
    spans: List[Tuple[int, int, str]], text_len: int
) -> List[Tuple[int, int, str]]:
    spans.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int, str]] = []
    cur_s = cur_e = None
    cur_t = ""
    for s, e, t in spans:
        if cur_s is None:
            cur_s, cur_e, cur_t = s, e, t
            continue
        if s <= cur_e:  # overlap
            cur_e = max(cur_e, e)
            if len(t) > len(cur_t):
                cur_t = t
        else:
            merged.append((cur_s, cur_e, cur_t))
            cur_s, cur_e, cur_t = s, e, t
    if cur_s is not None:
        merged.append((cur_s, cur_e, cur_t))

    final = [(max(0, s), min(text_len, e), t) for s, e, t in merged if e > s]
    return final or [(0, text_len, "FullDoc")]


# ================================================================
# Global, shared ChatOpenAI client
# ================================================================
LLM_CLIENT = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_retries=0,  # disable LC retries – handled by tenacity wrapper
    request_timeout=LlmChunkerConfig().request_timeout,
    api_key=os.getenv("OPENAI_API_KEY"),  # explicit for clarity
)


@retry(
    stop=stop_after_attempt(LlmChunkerConfig().max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def _call_llm(messages: List[Dict[str, str]]) -> str:
    """Call OpenAI chat completion with retries & timeout."""
    resp = LLM_CLIENT.invoke(messages)
    return resp.content if hasattr(resp, "content") else str(resp)


# ================================================================
# Segment a single *text* (LLM windows + merge)
# ================================================================
def _segment_text_with_llm(text: str, cfg: LlmChunkerConfig) -> List[Tuple[int, int, str]]:
    windows = _window_text(text, cfg.window_chars, cfg.overlap_chars)
    spans: List[Tuple[int, int, str]] = []

    for idx, (w_start, w_txt) in enumerate(windows, 1):
        try:
            raw = _call_llm(_build_messages(w_txt, cfg))
            local = _parse_segments(raw, len(w_txt))
        except Exception as e:
            print(f"[warn] window {idx} failed ({e}); using fallback span")
            local = [(0, len(w_txt), "FullSpan")]
        spans.extend((w_start + s, w_start + e, t) for s, e, t in local)

    return _merge_overlaps(spans, len(text))


# ================================================================
# Public API – chunk a single Document
# ================================================================
def chunk_doc_via_llm(doc: Document, cfg: Optional[LlmChunkerConfig] = None) -> List[Document]:
    cfg = cfg or LlmChunkerConfig()
    enc = _get_encoder(cfg.enc_name)
    text = doc.page_content

    segs = _segment_text_with_llm(text, cfg)
    out_docs: List[Document] = []

    for s, e, title in segs:
        seg_txt = text[s:e].strip()
        tok_count = _count_tokens(seg_txt, enc)

        if tok_count > cfg.max_tokens:
            # extremely rare: LLM ignored hard cap – split deterministically
            for rs, re in _split_segment_tokenwise(seg_txt, enc, cfg.max_tokens):
                sub_txt = seg_txt[rs:re].strip()
                md = dict(doc.metadata)
                md.update(
                    {
                        "source_start_char": s + rs,
                        "source_end_char": s + re,
                        "tokens": _count_tokens(sub_txt, enc),
                        "method": cfg.method_tag + ">token_fallback",
                        "parent_title": title if cfg.include_titles else "",
                    }
                )
                out_docs.append(Document(page_content=sub_txt, metadata=md))
        else:
            md = dict(doc.metadata)
            md.update(
                {
                    "start_char": s,
                    "end_char": e,
                    "tokens": tok_count,
                    "method": cfg.method_tag,
                    "title": title if cfg.include_titles else "",
                }
            )
            out_docs.append(Document(page_content=seg_txt, metadata=md))
    return out_docs


# ================================================================
# Public API – chunk a list of Documents
# ================================================================
def chunk_docs_via_llm(docs: List[Document], cfg: Optional[LlmChunkerConfig] = None) -> List[Document]:
    cfg = cfg or LlmChunkerConfig()
    out: List[Document] = []
    for idx, d in enumerate(docs, 1):
        out.extend(chunk_doc_via_llm(d, cfg))
        print(f"[info] processed doc {idx}/{len(docs)}")
    return out
