from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tiktoken

# ------------------------------------------------------------------
# LangChain Document import (compatible across versions)
# ------------------------------------------------------------------
try:
    from langchain_core.documents import Document
except ImportError:  # older LC
    from langchain.schema import Document  # type: ignore

from langchain_openai import ChatOpenAI


@dataclass
class LlmChunkerConfig:
    target_tokens: int = 700      # soft goal per semantic segment
    max_tokens: int = 1000        # hard cap for final chunk size
    hard_cap_tokens: int = 1200   # if LLM gives >this, we force split
    enc_name: str = "cl100k_base" # tokenizer for size checks
    window_chars: int = 15000     # doc window size sent to LLM
    overlap_chars: int = 800      # overlap to avoid boundary loss
    include_titles: bool = True   # attach LLM segment title in metadata
    method_tag: str = "semantic_llm"


def _get_encoder(enc_name: str):
    return tiktoken.get_encoding(enc_name)

def _count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def _window_text(text: str, win: int, overlap: int) -> List[Tuple[int, str]]:
    out = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + win)
        out.append((start, text[start:end]))
        if end == n:
            break
        start = max(0, end - overlap)
    return out



_PROMPT_TEMPLATE = """Segment the following TEXT into topically coherent sections for retrieval.

Return **ONLY** valid JSON of the form:
{{
  "segments": [
    {{"start_char": <int>, "end_char": <int>, "title": "<short title>"}},
    ...
  ]
}}

Rules:
- Offsets are 0-based character indexes **within the TEXT block below**.
- Cover the entire text; no gaps; no overlaps.
- Keep order.
- Aim for ~{target_tokens} tokens/segment (soft), max {max_tokens} tokens (hard), break further if >{hard_cap_tokens}.
- Do not cut mid-sentence unless required to stay under hard cap.
- Titles should be short summaries (<=10 words). If unknown, use an empty string.

TEXT:
\"\"\"{text}\"\"\""""


def _build_messages(
    text_block: str,
    target_tokens: int,
    max_tokens: int,
    hard_cap_tokens: int,
) -> List[Dict[str, str]]:
    """Return LangChain-style message list."""
    return [
        {"role": "system", "content": "You segment long text into coherent sections for retrieval-augmented generation."},
        {"role": "user", "content": _PROMPT_TEMPLATE.format(
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            hard_cap_tokens=hard_cap_tokens,
            text=text_block,
        )},
    ]


# ================================================================
# Robust JSON extraction
# ================================================================
_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)

def _extract_json(text: str) -> str:
    # Try full parse
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    # Try to grab first {...} blob
    m = _JSON_RE.search(text)
    if m:
        blob = m.group(0)
        try:
            json.loads(blob)
            return blob
        except Exception:
            pass
    # Give up; fallback sentinel
    return '{"segments":[{"start_char":0,"end_char":%d,"title":"FullSpan"}]}' % len(text)


# ================================================================
# Parse LLM output -> list of (start, end, title)
# ================================================================
def _parse_segments(raw_text: str, text_len: int) -> List[Tuple[int, int, str]]:
    blob = _extract_json(raw_text)
    try:
        data = json.loads(blob)
        segs = data.get("segments", [])
    except Exception:
        segs = [{"start_char": 0, "end_char": text_len, "title": "FullSpan"}]

    out: List[Tuple[int, int, str]] = []
    for seg in segs:
        try:
            s = int(seg["start_char"])
            e = int(seg["end_char"])
            t = str(seg.get("title", "")).strip()
        except Exception:
            continue
        # sanitize
        if s < 0: s = 0
        if e > text_len: e = text_len
        if e <= s: continue
        out.append((s, e, t))
    if not out:
        out = [(0, text_len, "FullSpan")]
    # ensure sorted / non-overlap
    out.sort(key=lambda x: x[0])
    cleaned = []
    last_end = 0
    for s, e, t in out:
        if s > last_end:
            # fill any gap
            cleaned.append((last_end, s, "AutoFill"))
        if s < last_end:
            s = last_end
        cleaned.append((s, e, t))
        last_end = e
    if last_end < text_len:
        cleaned.append((last_end, text_len, "AutoTail"))
    # merge adjacent Auto* if tiny
    merged: List[Tuple[int, int, str]] = []
    for s, e, t in cleaned:
        if merged and t.startswith("Auto") and merged[-1][2].startswith("Auto"):
            ps, pe, pt = merged[-1]
            merged[-1] = (ps, e, pt)
        else:
            merged.append((s, e, t))
    return merged


# ================================================================
# Fallback splitter for oversized segments (token aware char split)
# ================================================================
def _split_segment_tokenwise(
    text: str,
    enc,
    max_tokens: int,
) -> List[Tuple[int, int]]:
    """Return list of (start_char, end_char) splits so each part <= max_tokens."""
    tokens = enc.encode(text)
    spans: List[Tuple[int, int]] = []
    if len(tokens) <= max_tokens:
        return [(0, len(text))]

    # walk tokens -> chars
    # decode slices; this is slower but robust without external libs
    i = 0
    n = len(tokens)
    while i < n:
        j = min(n, i + max_tokens)
        sub_text = enc.decode(tokens[i:j])
        # find global char offsets by searching (costly but okay batch)
        # better: track char mapping; here simplified
        if i == 0:
            start_c = 0
        else:
            start_c = len(enc.decode(tokens[:i]))
        end_c = start_c + len(sub_text)
        spans.append((start_c, end_c))
        i = j
    return spans


# ================================================================
# Main per-text segmentation using LLM
# ================================================================
def _segment_text_with_llm(
    text: str,
    cfg: LlmChunkerConfig,
) -> List[Tuple[int, int, str]]:
    """Call LLM over windows; return merged global segments."""
    windows = _window_text(text, cfg.window_chars, cfg.overlap_chars)
    segs_global: List[Tuple[int, int, str]] = []

    for win_start, win_txt in windows:
        msgs = _build_messages(
            win_txt,
            target_tokens=cfg.target_tokens,
            max_tokens=cfg.max_tokens,
            hard_cap_tokens=cfg.hard_cap_tokens,
        )
        # LangChain chat models: .invoke(messages) -> BaseMessage
        llm = ChatOpenAI()
        resp = llm.invoke(msgs)
        raw = getattr(resp, "content", resp)  # try to be flexible
        local_segs = _parse_segments(raw, len(win_txt))
        # map to global char indexes
        for s, e, t in local_segs:
            segs_global.append((win_start + s, win_start + e, t))

    # merge overlapping global segs
    segs_global.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int, str]] = []
    cur_s = cur_e = None
    cur_t = ""
    for s, e, t in segs_global:
        if cur_s is None:
            cur_s, cur_e, cur_t = s, e, t
            continue
        if s <= cur_e:  # overlap
            cur_e = max(cur_e, e)
            # keep more descriptive title
            if len(t) > len(cur_t):
                cur_t = t
        else:
            merged.append((cur_s, cur_e, cur_t))
            cur_s, cur_e, cur_t = s, e, t
    if cur_s is not None:
        merged.append((cur_s, cur_e, cur_t))

    # clamp to doc bounds
    doc_len = len(text)
    clamped = []
    for s, e, t in merged:
        s = max(0, min(s, doc_len))
        e = max(0, min(e, doc_len))
        if e > s:
            clamped.append((s, e, t))
    if not clamped:
        clamped = [(0, doc_len, "FullDoc")]

    return clamped


# ================================================================
# Public API: chunk a single Document
# ================================================================
def chunk_doc_via_llm(
    doc: Document,
    cfg: Optional[LlmChunkerConfig] = None,
) -> List[Document]:
    cfg = cfg or LlmChunkerConfig()
    enc = _get_encoder(cfg.enc_name)
    text = doc.page_content

    segs = _segment_text_with_llm(text, cfg)

    out_docs: List[Document] = []
    for (s, e, title) in segs:
        seg_txt = text[s:e].strip()
        tok_count = _count_tokens(seg_txt, enc)

        if tok_count > cfg.max_tokens:
            rel_spans = _split_segment_tokenwise(seg_txt, enc, cfg.max_tokens)
            for rs, re in rel_spans:
                sub_txt = seg_txt[rs:re].strip()
                sub_tokens = _count_tokens(sub_txt, enc)
                md = dict(doc.metadata)
                md.update({
                    "source_start_char": s + rs,
                    "source_end_char": s + re,
                    "tokens": sub_tokens,
                    "method": cfg.method_tag + ">token_fallback",
                })
                if cfg.include_titles:
                    md["parent_title"] = title
                out_docs.append(Document(page_content=sub_txt, metadata=md))
        else:
            md = dict(doc.metadata)
            md.update({
                "start_char": s,
                "end_char": e,
                "tokens": tok_count,
                "method": cfg.method_tag,
            })
            if cfg.include_titles:
                md["title"] = title
            out_docs.append(Document(page_content=seg_txt, metadata=md))

    return out_docs


def chunk_docs_via_llm(
    docs: List[Document],
) -> List[Document]:
    cfg = LlmChunkerConfig()
    out: List[Document] = []
    count = 0
    for d in docs:
        doc_chunked = chunk_doc_via_llm(d, cfg=cfg)
        out.extend(doc_chunked)
        count = count+1
        print(count)
    return out
