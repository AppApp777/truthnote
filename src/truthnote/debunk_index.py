# src/truthnote/debunk_index.py
"""
Official debunk evidence retriever for TruthNote.

Design:
- Reuse public_board.load_crawler_rumors(); do not re-read JSONL here.
- Build one lazy in-memory lexical index over official debunk rows.
- Retrieval is no-LLM / offline-safe: Chinese char n-gram + BM25-ish scoring.
- A retrieved row is NOT a verdict. It becomes Evidence only after same-claim guards pass.

Public API:
    retrieve_debunk_candidates(claim_text, top_k=3) -> list[DebunkCandidate]
    verify_same_claim(claim_text, candidate) -> SameClaimResult
    confirmed_candidate_to_evidence(candidate, result=None) -> Evidence
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from .memory import _claim_fingerprint, _entities_compatible, _extract_entities
from .public_board import load_crawler_rumors
from .schemas import Evidence, SourceType

SameClaimLabel = Literal[
    "same_claim",
    "same_topic_different_claim",
    "opposite_claim",
    "no_match",
]


# ──────────────────────────────────────────────────────────────────────────────
# Conservative thresholds.
#
# Retrieval is allowed to be recall-oriented. Verification is precision-oriented.
# Only label == "same_claim" may be converted into Evidence.
# ──────────────────────────────────────────────────────────────────────────────

RETRIEVAL_MIN_SCORE = 0.18

SAME_TOPIC_MIN_SCORE = 0.45
SAME_CLAIM_MIN_SCORE = 0.72
SAME_CLAIM_MIN_PAIR_SIM = 0.50

MAIN_ENTITY_OVERLAP_MIN = 0.25

BM25_K1 = 1.4
BM25_B = 0.75


class DebunkCandidate(BaseModel):
    """A candidate retrieved from the official debunk DB. This is NOT a verdict."""

    item_id: str
    claim_text: str
    verdict: str = ""

    source: str = ""
    url: str = ""
    title: str = ""
    snippet: str = ""
    category: str = ""
    published_date: str = ""

    lexical_score: float = 0.0
    bm25_score: float = 0.0
    ngram_score: float = 0.0
    token_score: float = 0.0
    entity_score: float = 0.0
    exact_match: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)


class SameClaimResult(BaseModel):
    """Result of strict same-claim verification."""

    label: SameClaimLabel
    score: float

    candidate_id: str = ""
    candidate_url: str = ""
    matched_claim: str = ""

    reasons: list[str] = Field(default_factory=list)
    passed_guards: list[str] = Field(default_factory=list)
    failed_guards: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _IndexedRumor:
    item_id: str
    claim_text: str
    verdict: str
    source: str
    url: str
    category: str
    published_date: str

    compact: str
    fingerprint: str
    terms: tuple[str, ...]
    term_counts: Counter[str]
    ngrams: frozenset[str]
    entities: frozenset[str]
    doc_len: int


class _DebunkIndex:
    def __init__(self, docs: list[_IndexedRumor]):
        self.docs = docs
        self.n_docs = len(docs)
        self.avgdl = sum(d.doc_len for d in docs) / self.n_docs if self.n_docs else 1.0

        self.df: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_id, doc in enumerate(docs):
            for term, tf in doc.term_counts.items():
                self.df[term] += 1
                self.postings[term].append((doc_id, tf))

    def idf(self, term: str) -> float:
        # BM25 IDF with +1 smoothing.
        df = self.df.get(term, 0)
        return math.log(1.0 + ((self.n_docs - df + 0.5) / (df + 0.5)))

    def search(self, claim_text: str, top_k: int = 3) -> list[DebunkCandidate]:
        if top_k <= 0:
            return []

        query = (claim_text or "").strip()
        if not query or not self.docs:
            return []

        q_terms = _lexical_terms(query)
        if not q_terms:
            return []

        q_counts = Counter(q_terms)
        q_ngrams = _char_ngrams(query)
        q_entities = frozenset(_extract_entities(query))
        q_compact = _compact_text(query)
        q_fingerprint = _safe_fingerprint(query)

        raw_scores: defaultdict[int, float] = defaultdict(float)

        # BM25-ish candidate generation through inverted index.
        for term in q_counts:
            idf = self.idf(term)
            for doc_id, tf in self.postings.get(term, []):
                doc = self.docs[doc_id]
                denom = tf + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * (doc.doc_len / max(self.avgdl, 1e-9))
                )
                raw_scores[doc_id] += idf * ((tf * (BM25_K1 + 1.0)) / denom)

        # Cheap exact / containment boost. Scanning ~5,000 docs is trivial and
        # avoids missing short exact matches due to tokenization.
        if q_compact:
            for doc_id, doc in enumerate(self.docs):
                if not doc.compact:
                    continue
                if q_fingerprint and q_fingerprint == doc.fingerprint:
                    raw_scores[doc_id] += 50.0
                else:
                    contained, ratio = _compact_containment(q_compact, doc.compact)
                    if contained and ratio >= 0.55:
                        raw_scores[doc_id] += 8.0 + (4.0 * ratio)

        if not raw_scores:
            return []

        scored: list[tuple[float, DebunkCandidate]] = []
        for doc_id, raw_bm25 in raw_scores.items():
            doc = self.docs[doc_id]
            candidate, final_score = self._candidate_from_doc(
                doc=doc,
                rank_doc_id=doc_id,
                query=query,
                q_compact=q_compact,
                q_fingerprint=q_fingerprint,
                q_ngrams=q_ngrams,
                q_entities=q_entities,
                raw_bm25=raw_bm25,
            )
            if final_score >= RETRIEVAL_MIN_SCORE:
                scored.append((final_score, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def _candidate_from_doc(
        self,
        *,
        doc: _IndexedRumor,
        rank_doc_id: int,
        query: str,
        q_compact: str,
        q_fingerprint: str,
        q_ngrams: frozenset[str],
        q_entities: frozenset[str],
        raw_bm25: float,
    ) -> tuple[DebunkCandidate, float]:
        bm25_norm = raw_bm25 / (raw_bm25 + 8.0) if raw_bm25 > 0 else 0.0

        jaccard, containment = _ngram_jaccard_and_containment(q_ngrams, doc.ngrams)
        ngram_score = (0.35 * jaccard) + (0.65 * containment)

        entity_score = _entity_overlap_score(q_entities, doc.entities)

        exact_match = bool(q_fingerprint and q_fingerprint == doc.fingerprint)
        contained, containment_ratio = _compact_containment(q_compact, doc.compact)

        final_score = 0.40 * bm25_norm + 0.35 * containment + 0.20 * jaccard + 0.05 * entity_score

        if exact_match:
            final_score = max(final_score, 0.99)
        elif contained and containment_ratio >= 0.65:
            final_score = max(final_score, 0.84 + 0.12 * containment_ratio)

        # Retrieval should remain recall-oriented, but obvious number/date
        # conflicts should be dampened so they do not dominate top-k.
        if not _entities_compatible(query, doc.claim_text):
            final_score *= 0.72

        final_score = _clip01(final_score)

        candidate = DebunkCandidate(
            item_id=doc.item_id,
            claim_text=doc.claim_text,
            verdict=doc.verdict,
            source=doc.source,
            url=doc.url,
            title=_make_title(doc.claim_text),
            snippet=f"官方辟谣库收录：{doc.claim_text[:180]}",
            category=doc.category,
            published_date=doc.published_date,
            lexical_score=final_score,
            bm25_score=_clip01(bm25_norm),
            ngram_score=_clip01(ngram_score),
            token_score=_clip01(containment),
            entity_score=_clip01(entity_score),
            exact_match=exact_match,
            metadata={
                "doc_id": rank_doc_id,
                "raw_bm25": raw_bm25,
                "ngram_jaccard": jaccard,
                "ngram_containment": containment,
                "compact_containment": contained,
                "compact_containment_ratio": containment_ratio,
            },
        )
        return candidate, final_score


# ──────────────────────────────────────────────────────────────────────────────
# Lazy cached index construction.
# ──────────────────────────────────────────────────────────────────────────────

_index_cache: _DebunkIndex | None = None
_index_lock = threading.Lock()


def get_debunk_index(*, force: bool = False) -> _DebunkIndex:
    """Build the official debunk index once, lazily, using load_crawler_rumors()."""
    global _index_cache

    if _index_cache is not None and not force:
        return _index_cache

    with _index_lock:
        if _index_cache is None or force:
            rows = load_crawler_rumors(force=force)
            docs: list[_IndexedRumor] = []

            for item in rows:
                claim_text = (getattr(item, "claim_text", "") or "").strip()
                if not claim_text:
                    continue

                evidence_urls = getattr(item, "evidence_urls", None) or []
                url = evidence_urls[0] if evidence_urls else ""

                source = (getattr(item, "reported_to", "") or "").strip()
                item_id = (getattr(item, "item_id", "") or "").strip()
                category = (getattr(item, "category", "") or "").strip()
                published_date = _date_to_str(getattr(item, "created_at", ""))

                verdict = _verdict_to_text(getattr(item, "verdict", ""))

                terms = tuple(_lexical_terms(claim_text))
                if not terms:
                    continue

                term_counts = Counter(terms)
                docs.append(
                    _IndexedRumor(
                        item_id=item_id or _safe_fingerprint(claim_text)[:16],
                        claim_text=claim_text,
                        verdict=verdict,
                        source=source or "官方辟谣平台",
                        url=url,
                        category=category,
                        published_date=published_date,
                        compact=_compact_text(claim_text),
                        fingerprint=_safe_fingerprint(claim_text),
                        terms=terms,
                        term_counts=term_counts,
                        ngrams=_char_ngrams(claim_text),
                        entities=frozenset(_extract_entities(claim_text)),
                        doc_len=max(1, len(terms)),
                    )
                )

            _index_cache = _DebunkIndex(docs)

    return _index_cache


def clear_debunk_index_cache() -> None:
    """Useful for tests."""
    global _index_cache
    with _index_lock:
        _index_cache = None


def retrieve_debunk_candidates(
    claim_text: str,
    top_k: int = 3,
) -> list[DebunkCandidate]:
    """
    Retrieve candidate official debunks for the claim.

    This is intentionally lexical / offline-safe. It does not call embeddings,
    LLMs, search APIs, or any network-dependent component.
    """
    return get_debunk_index().search(claim_text, top_k=top_k)


# ──────────────────────────────────────────────────────────────────────────────
# Same-claim verification.
# ──────────────────────────────────────────────────────────────────────────────


def verify_same_claim(
    claim_text: str,
    candidate: DebunkCandidate,
) -> SameClaimResult:
    """
    Conservative same-claim verifier.

    Guard order:
    1. Empty / exact fingerprint.
    2. Cheap lexical relatedness gate.
    3. Number/date guard via memory._entities_compatible().
    4. Negation-polarity guard.
    5. Status/modality guard.
    6. Main-entity overlap guard.
    7. Final precision-weighted score threshold.

    Only label == "same_claim" may become Evidence.
    """
    q = (claim_text or "").strip()
    c = (candidate.claim_text or candidate.title or "").strip()

    reasons: list[str] = []
    passed: list[str] = []
    failed: list[str] = []

    def finish(label: SameClaimLabel, score: float) -> SameClaimResult:
        return SameClaimResult(
            label=label,
            score=_clip01(score),
            candidate_id=candidate.item_id,
            candidate_url=candidate.url,
            matched_claim=c,
            reasons=reasons,
            passed_guards=passed,
            failed_guards=failed,
        )

    if not q or not c:
        failed.append("empty_claim")
        reasons.append("输入 claim 或候选 claim 为空。")
        return finish("no_match", 0.0)

    q_fp = _safe_fingerprint(q)
    c_fp = _safe_fingerprint(c)

    if q_fp and q_fp == c_fp:
        passed.append("exact_fingerprint")
        reasons.append("规范化指纹完全一致。")
        return finish("same_claim", 0.99)

    pair_score, pair_jaccard, pair_containment = _pair_similarity(q, c)
    retrieval_score = max(
        float(candidate.lexical_score or 0.0),
        float(candidate.ngram_score or 0.0),
        pair_score,
    )

    reasons.append(
        f"lexical_pair={pair_score:.2f}, "
        f"jaccard={pair_jaccard:.2f}, "
        f"containment={pair_containment:.2f}, "
        f"retrieval={retrieval_score:.2f}"
    )

    if retrieval_score < RETRIEVAL_MIN_SCORE and pair_score < 0.28:
        failed.append("lexical_gate")
        reasons.append("词面相关度过低，判为 no_match。")
        return finish("no_match", max(retrieval_score, pair_score))

    passed.append("lexical_gate")

    # Reuse existing number/date guards from memory.py.
    if not _entities_compatible(q, c):
        q_ents = sorted(_extract_entities(q))
        c_ents = sorted(_extract_entities(c))
        failed.append("number_or_date_conflict")
        reasons.append(f"数字/日期实体冲突：query={q_ents}, candidate={c_ents}")
        label: SameClaimLabel = (
            "same_topic_different_claim" if max(retrieval_score, pair_score) >= 0.32 else "no_match"
        )
        return finish(label, min(0.66, max(retrieval_score, pair_score)))

    passed.append("number_date_guard")

    polarity_conflict, polarity_reason = _polarity_conflict(q, c)
    if polarity_conflict:
        failed.append("negation_polarity_conflict")
        reasons.append(polarity_reason)
        return finish("opposite_claim", min(0.80, max(retrieval_score, pair_score)))

    passed.append("negation_polarity_guard")

    status_conflict, status_reason = _status_conflict(q, c)
    if status_conflict:
        failed.append("status_or_modality_conflict")
        reasons.append(status_reason)
        return finish(
            "same_topic_different_claim",
            min(0.76, max(retrieval_score, pair_score)),
        )

    passed.append("status_modality_guard")

    main_ok, main_score, main_reason = _main_entity_assessment(q, c)
    if not main_ok:
        failed.append("main_entity_conflict")
        reasons.append(main_reason)
        return finish(
            "same_topic_different_claim",
            min(0.74, max(retrieval_score, pair_score)),
        )

    passed.append("main_entity_guard")
    reasons.append(main_reason)

    final_score = 0.65 * pair_score + 0.20 * retrieval_score + 0.15 * main_score

    q_compact = _compact_text(q)
    c_compact = _compact_text(c)
    contained, containment_ratio = _compact_containment(q_compact, c_compact)
    if contained and containment_ratio >= 0.78:
        final_score = max(final_score, 0.82)

    final_score = _clip01(final_score)

    if final_score >= SAME_CLAIM_MIN_SCORE and pair_score >= SAME_CLAIM_MIN_PAIR_SIM:
        passed.append("same_claim_threshold")
        reasons.append(
            f"同命题阈值通过：final={final_score:.2f}, "
            f"pair={pair_score:.2f}, main_entity={main_score:.2f}"
        )
        return finish("same_claim", final_score)

    if final_score >= SAME_TOPIC_MIN_SCORE:
        failed.append("same_claim_threshold")
        reasons.append(f"仅达到同主题强度，未达到同命题阈值：final={final_score:.2f}")
        return finish("same_topic_different_claim", final_score)

    failed.append("same_claim_threshold")
    reasons.append(f"未达到同主题/同命题阈值：final={final_score:.2f}")
    return finish("no_match", final_score)


# 官方裁决里属于「证伪/辟谣」家族的关键词。只有命中这些，命中候选才被当作辟谣证据
# （supports_claim=False + 辟谣措辞）。非证伪定性（如「无法核实」）走中性分支，不注入辟谣信号。
_DEBUNK_VERDICT_KEYWORDS = (
    "谣",
    "假",
    "不实",
    "虚假",
    "失实",
    "伪",
    "辟谣",
    "误导",
    "造谣",
    "捏造",
    "夸大",
)


def _is_debunk_verdict(verdict_text: str) -> bool:
    """官方裁决是否属于「证伪/辟谣」家族。空裁决按辟谣库性质默认视为辟谣。"""
    v = (verdict_text or "").strip()
    if not v:
        return True
    return any(k in v for k in _DEBUNK_VERDICT_KEYWORDS)


def confirmed_candidate_to_evidence(
    candidate: DebunkCandidate,
    result: SameClaimResult | None = None,
) -> Evidence:
    """
    Convert a confirmed same-claim candidate into project Evidence.

    This must only be called after verify_same_claim(...).label == "same_claim".

    裁决分流（adversarial-review HIGH 修）：只有「证伪/辟谣」家族的官方裁决才包成
    辟谣证据（supports_claim=False + 辟谣措辞 + S-辟谣库命中），让下游正确识别为
    debunk 信号。库里少量「无法核实」等非证伪定性若被同命题命中，**不能**伪装成辟谣
    （否则会把一条「官方都说查不到」的消息错误地推向「假」）——走中性分支，去掉一切
    辟谣措辞，避免触发下游 `_has_debunk_signal`。
    """
    if result is not None and result.label != "same_claim":
        raise ValueError(
            "confirmed_candidate_to_evidence() requires SameClaimResult.label == 'same_claim'"
        )

    raw_verdict = (candidate.verdict or "").strip()
    verdict_text = raw_verdict or "官方辟谣"
    is_debunk = _is_debunk_verdict(raw_verdict)
    score_text = f"；同命题核对分数：{result.score:.2f}" if result is not None else ""

    if is_debunk:
        snippet = (
            f"官方辟谣库命中同一命题：{candidate.claim_text[:220]}"
            f"；官方结论：{verdict_text}"
            f"{score_text}"
        )
        credibility = "S-辟谣库命中"
    else:
        # 中性分支：措辞刻意避开 _DEBUNK_KEYWORDS（辟谣/不实/谣言/未经证实…），
        # 不让一条非证伪定性被误判成 debunk 信号。
        snippet = (
            f"官方核查库收录同一命题：{candidate.claim_text[:220]}"
            f"；官方结论：{verdict_text}（非证伪定性，仅供参考）"
            f"{score_text}"
        )
        credibility = "S-官方核查库"

    return Evidence(
        source=candidate.source or "官方辟谣平台",
        url=candidate.url or "",
        title=candidate.title or _make_title(candidate.claim_text),
        snippet=snippet[:500],
        credibility=credibility,
        supports_claim=False,
        source_tag="official_debunk_index",
        source_type=SourceType.FACT_CHECK_ORG,
        authority_score=0.90,
        published_date=candidate.published_date or "",
        is_original_source=False,
        # 可视化采信理由：把同命题核对结果作为结构化标签带出，供前端渲染「为什么采信」。
        # 只有 same_claim 能走到这里（上面已校验 result.label）。无 result 时不臆造标签，
        # 留空——结构化标签必须有同命题核对结果背书（adversarial-review LOW）。
        match_label=result.label if result is not None else "",
        match_score=round(result.score, 2) if result is not None else None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Lexical retrieval helpers.
# ──────────────────────────────────────────────────────────────────────────────

_KEEP_RE = re.compile(r"[一-鿿A-Za-z0-9%％.\-/年月日号]+")
_ALNUM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|\d+(?:\.\d+)?[%％]?")

_CHINESE_PUNCT_RE = re.compile(r"[，。！？、；：“”‘’（）《》【】「」『』—…·\s]+")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def _compact_text(text: str) -> str:
    t = _normalize(text)
    t = re.sub(r"https?://\S+", "", t)
    parts = _KEEP_RE.findall(t)
    return "".join(parts)


def _char_ngrams(text: str, ns: tuple[int, ...] = (2, 3)) -> frozenset[str]:
    compact = _compact_text(text)
    if not compact:
        return frozenset()

    grams: list[str] = []
    for n in ns:
        if len(compact) >= n:
            grams.extend(compact[i : i + n] for i in range(len(compact) - n + 1))

    if not grams:
        grams.append(compact)

    return frozenset(grams)


def _lexical_terms(text: str) -> list[str]:
    """
    Chinese-tokenizer-free terms:
    - char bigrams/trigrams over compact Chinese text;
    - extracted number/date entities from memory._extract_entities();
    - Latin / numeric tokens.
    """
    terms: list[str] = list(_char_ngrams(text))

    for ent in _extract_entities(text):
        if ent:
            terms.append(f"ent:{ent}")

    normalized = _normalize(text)
    for m in _ALNUM_RE.finditer(normalized):
        tok = m.group().strip().lower()
        if tok:
            terms.append(tok)

    return terms


def _ngram_jaccard_and_containment(
    a: frozenset[str],
    b: frozenset[str],
) -> tuple[float, float]:
    if not a or not b:
        return 0.0, 0.0

    inter = len(a & b)
    if inter <= 0:
        return 0.0, 0.0

    jaccard = inter / max(1, len(a | b))
    containment = inter / max(1, min(len(a), len(b)))
    return _clip01(jaccard), _clip01(containment)


def _pair_similarity(a: str, b: str) -> tuple[float, float, float]:
    a_grams = _char_ngrams(a)
    b_grams = _char_ngrams(b)
    jaccard, containment = _ngram_jaccard_and_containment(a_grams, b_grams)

    score = (0.35 * jaccard) + (0.65 * containment)
    return _clip01(score), jaccard, containment


def _compact_containment(a_compact: str, b_compact: str) -> tuple[bool, float]:
    if not a_compact or not b_compact:
        return False, 0.0

    if a_compact == b_compact:
        return True, 1.0

    if a_compact in b_compact or b_compact in a_compact:
        ratio = min(len(a_compact), len(b_compact)) / max(
            1,
            max(len(a_compact), len(b_compact)),
        )
        return True, _clip01(ratio)

    return False, 0.0


def _entity_overlap_score(
    q_entities: frozenset[str],
    doc_entities: frozenset[str],
) -> float:
    if not q_entities and not doc_entities:
        return 0.5
    if not q_entities or not doc_entities:
        return 0.45

    inter = q_entities & doc_entities
    if inter:
        return _clip01(len(inter) / max(1, max(len(q_entities), len(doc_entities))))

    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Same-claim guard helpers.
# ──────────────────────────────────────────────────────────────────────────────

_LOCATION_TERMS = (
    # Municipalities / provinces / regions.
    "北京",
    "北京市",
    "上海",
    "上海市",
    "天津",
    "天津市",
    "重庆",
    "重庆市",
    "河北",
    "山西",
    "辽宁",
    "吉林",
    "黑龙江",
    "江苏",
    "浙江",
    "安徽",
    "福建",
    "江西",
    "山东",
    "河南",
    "湖北",
    "湖南",
    "广东",
    "海南",
    "四川",
    "贵州",
    "云南",
    "陕西",
    "甘肃",
    "青海",
    "台湾",
    "内蒙古",
    "广西",
    "西藏",
    "宁夏",
    "新疆",
    "香港",
    "澳门",
    # Common cities.
    "广州",
    "深圳",
    "杭州",
    "南京",
    "苏州",
    "成都",
    "武汉",
    "西安",
    "长沙",
    "郑州",
    "青岛",
    "厦门",
    "福州",
    "宁波",
    "合肥",
    "济南",
    "昆明",
    "南宁",
    "南昌",
    "贵阳",
    "兰州",
    "太原",
    "石家庄",
    "沈阳",
    "长春",
    "哈尔滨",
)

_SALIENT_TERMS = (
    "疫苗",
    "病毒",
    "新冠",
    "流感",
    "癌症",
    "不孕",
    "食品",
    "药品",
    "保健品",
    "补贴",
    "养老金",
    "医保",
    "社保",
    "房产税",
    "个税",
    "税费",
    "新规",
    "政策",
    "通知",
    "地震",
    "洪水",
    "台风",
    "暴雨",
    "学校",
    "考试",
    "高考",
    "银行",
    "贷款",
    "存款",
    "诈骗",
    "火灾",
    "爆炸",
    "交通",
    "限行",
)

_ORG_RE = re.compile(
    r"[一-鿿A-Za-z0-9]{2,30}"
    r"(?:部|委|局|厅|院|所|中心|协会|公司|集团|医院|学校|大学|平台|政府|公安|法院|检察院|银行)"
)

_STATUS_PATTERNS: dict[str, tuple[str, ...]] = {
    "implemented": (
        "已实施",
        "已经实施",
        "正式实施",
        "开始实施",
        "施行",
        "生效",
        "落地",
        "执行",
    ),
    "consultation": (
        "征求意见",
        "公开征求",
        "征求公众意见",
        "意见稿",
        "草案",
        "拟",
        "拟定",
        "拟出台",
    ),
    "future": (
        "将实施",
        "即将实施",
        "计划实施",
        "预计实施",
        "将发布",
        "即将发布",
    ),
    "not_implemented": (
        "未实施",
        "尚未实施",
        "没有实施",
        "暂未实施",
    ),
    "cancelled": (
        "取消",
        "停止",
        "暂停",
        "叫停",
    ),
}

_STATUS_CONFLICTS = {
    frozenset(("implemented", "consultation")),
    frozenset(("implemented", "future")),
    frozenset(("implemented", "not_implemented")),
    frozenset(("implemented", "cancelled")),
    frozenset(("future", "cancelled")),
}

_NEGATIVE_RE = re.compile(
    r"(?:"
    r"不会|不能|无法|不可能|不是|并非|并不|没有|没法|"
    r"未曾|未发现|未实施|未发布|未开始|"
    r"无证据|无效|无需|无须|不存在|"
    r"不(?:会|能|是|属实|存在|需要|再|可|可以|应|宜|建议|导致|引发|造成|发放|实施|征收)"
    r")"
)

_AFFIRMATIVE_WORDS = (
    "会",
    "是",
    "有",
    "存在",
    "已经",
    "已",
    "导致",
    "引发",
    "造成",
    "发放",
    "实施",
    "征收",
    "发布",
    "开始",
)

_POLARITY_PREDICATES = (
    "导致",
    "引发",
    "造成",
    "是",
    "有",
    "存在",
    "实施",
    "发放",
    "征收",
    "取消",
    "发布",
    "开始",
    # 模态谓词：真实库实测发现「X可以/能/可 Y」与「X不可以/不能/不可 Y」
    # 曾被误判 same_claim（否定翻转漏洞）。逐 occurrence 局部否定判定，
    # 只有同一谓词在两侧极性相反才冲突，故对模板一致的近义副本不误伤（仍判 same_claim）。
    "可以",
    "能",
    "可",
)


def _extract_main_entities(text: str) -> set[str]:
    compact = _compact_text(text)
    normalized = _normalize(text)

    entities: set[str] = set()

    for loc in _LOCATION_TERMS:
        if loc and loc in compact:
            # Normalize 北京市 -> 北京 style through keeping raw still okay;
            # conflict guard handles disjoint location mentions.
            entities.add(f"loc:{loc}")

    for m in _ORG_RE.finditer(compact):
        org = m.group().strip()
        if len(org) >= 3:
            entities.add(f"org:{org[:30]}")

    for m in _ALNUM_RE.finditer(normalized):
        tok = m.group().lower()
        if len(tok) >= 2 and not tok.isdigit():
            entities.add(f"tok:{tok}")

    for term in _SALIENT_TERMS:
        if term in compact:
            entities.add(f"term:{term}")

    return entities


def _by_prefix(items: set[str], prefix: str) -> set[str]:
    return {x.removeprefix(prefix) for x in items if x.startswith(prefix)}


def _main_entity_assessment(query_text: str, candidate_text: str) -> tuple[bool, float, str]:
    q_main = _extract_main_entities(query_text)
    c_main = _extract_main_entities(candidate_text)

    q_locs = _by_prefix(q_main, "loc:")
    c_locs = _by_prefix(c_main, "loc:")

    # Hard guard: same template but different jurisdiction is a classic false positive.
    if q_locs and c_locs and q_locs.isdisjoint(c_locs):
        return (
            False,
            0.0,
            f"地域实体冲突：query={sorted(q_locs)}, candidate={sorted(c_locs)}",
        )

    q_orgs = _by_prefix(q_main, "org:")
    c_orgs = _by_prefix(c_main, "org:")

    if q_orgs and c_orgs and q_orgs.isdisjoint(c_orgs):
        return (
            False,
            0.0,
            f"机构实体冲突：query={sorted(q_orgs)}, candidate={sorted(c_orgs)}",
        )

    # Neutral if neither side exposes reliable main entities.
    if not q_main or not c_main:
        return True, 1.0, "主实体信息不足，主实体 guard 中性通过。"

    inter = q_main & c_main
    score = len(inter) / max(1, min(len(q_main), len(c_main)))

    if score >= MAIN_ENTITY_OVERLAP_MIN:
        return (
            True,
            _clip01(score),
            f"主实体重叠通过：overlap={sorted(inter)[:6]}, score={score:.2f}",
        )

    q_named = {x for x in q_main if x.startswith(("loc:", "org:", "tok:"))}
    c_named = {x for x in c_main if x.startswith(("loc:", "org:", "tok:"))}

    if q_named and c_named:
        return (
            False,
            _clip01(score),
            f"主实体重叠不足：query={sorted(q_main)[:8]}, candidate={sorted(c_main)[:8]}",
        )

    # If only broad salient terms exist, do not hard-fail. Let final score decide.
    weak_reason = (
        f"仅有弱主题词重叠，主实体 guard 弱通过："
        f"query={sorted(q_main)[:8]}, candidate={sorted(c_main)[:8]}"
    )
    return (True, 0.50, weak_reason)


def _status_signature(text: str) -> set[str]:
    compact = _compact_text(text)
    out: set[str] = set()

    for label, patterns in _STATUS_PATTERNS.items():
        if any(p in compact for p in patterns):
            out.add(label)

    return out


def _status_conflict(query_text: str, candidate_text: str) -> tuple[bool, str]:
    q_status = _status_signature(query_text)
    c_status = _status_signature(candidate_text)

    if not q_status or not c_status:
        return False, "状态/时态信息不足，status guard 中性通过。"

    for q in q_status:
        for c in c_status:
            if frozenset((q, c)) in _STATUS_CONFLICTS:
                return (
                    True,
                    f"状态/时态冲突：query={sorted(q_status)}, candidate={sorted(c_status)}",
                )

    return False, f"状态/时态兼容：query={sorted(q_status)}, candidate={sorted(c_status)}"


def _is_locally_negated(compact_text: str, start: int, window: int = 4) -> bool:
    prefix = compact_text[max(0, start - window) : start]
    return any(marker in prefix for marker in ("不", "没", "未", "无", "非"))


def _has_explicit_negative(text: str) -> bool:
    return bool(_NEGATIVE_RE.search(_compact_text(text)))


def _has_explicit_affirmative(text: str) -> bool:
    compact = _compact_text(text)

    for word in _AFFIRMATIVE_WORDS:
        for m in re.finditer(re.escape(word), compact):
            # Avoid obvious non-polar words like 会议.
            if word == "会":
                nxt = compact[m.end() : m.end() + 1]
                if nxt in {"议", "员", "场"}:
                    continue

            # Avoid 有关 being treated as existential 有.
            if word == "有":
                nxt = compact[m.end() : m.end() + 1]
                if nxt == "关":
                    continue

            if not _is_locally_negated(compact, m.start()):
                return True

    return False


def _predicate_polarities(text: str) -> dict[str, set[str]]:
    compact = _compact_text(text)
    out: dict[str, set[str]] = defaultdict(set)

    for pred in _POLARITY_PREDICATES:
        for m in re.finditer(re.escape(pred), compact):
            if pred == "有":
                nxt = compact[m.end() : m.end() + 1]
                if nxt == "关":
                    continue

            pol = "neg" if _is_locally_negated(compact, m.start()) else "pos"
            out[pred].add(pol)

    return dict(out)


def _overall_polarity(text: str) -> str:
    neg = _has_explicit_negative(text)
    pos = _has_explicit_affirmative(text)

    if neg and not pos:
        return "negative"
    if pos and not neg:
        return "affirmative"
    if neg and pos:
        return "mixed"
    return "unknown"


def _polarity_sets_conflict(a: set[str], b: set[str]) -> bool:
    return (a == {"neg"} and b == {"pos"}) or (a == {"pos"} and b == {"neg"})


def _polarity_conflict(query_text: str, candidate_text: str) -> tuple[bool, str]:
    q_pred = _predicate_polarities(query_text)
    c_pred = _predicate_polarities(candidate_text)

    for pred in sorted(set(q_pred) & set(c_pred)):
        if _polarity_sets_conflict(q_pred[pred], c_pred[pred]):
            return (
                True,
                f"否定极性冲突：predicate={pred}, query={q_pred[pred]}, candidate={c_pred[pred]}",
            )

    q_kind = _overall_polarity(query_text)
    c_kind = _overall_polarity(candidate_text)

    if {q_kind, c_kind} == {"negative", "affirmative"}:
        return (
            True,
            f"整体否定极性冲突：query={q_kind}, candidate={c_kind}",
        )

    return False, f"否定极性兼容：query={q_kind}, candidate={c_kind}"


# ──────────────────────────────────────────────────────────────────────────────
# Misc helpers.
# ──────────────────────────────────────────────────────────────────────────────


def _safe_fingerprint(text: str) -> str:
    try:
        return _claim_fingerprint(text)
    except Exception:
        return _compact_text(text)


def _date_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        return str(value.isoformat())[:10]
    return str(value)


def _verdict_to_text(value: Any) -> str:
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    if enum_value:
        return str(enum_value)
    return str(value)


def _make_title(claim_text: str) -> str:
    t = _CHINESE_PUNCT_RE.sub(" ", claim_text or "").strip()
    return t[:80] or "官方辟谣库命中"


def _clip01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0
