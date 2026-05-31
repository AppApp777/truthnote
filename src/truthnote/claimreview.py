"""ClaimReview JSON-LD 导出。

标准 schema.org ClaimReview 格式，可对接任何平台（360 搜索生态、Google 等）。
"""

from __future__ import annotations

from datetime import datetime

from .schemas import ClaimVerification, Verdict, VerifyResponse

_VERDICT_TO_RATING: dict[Verdict, dict] = {
    Verdict.FALSE: {
        "ratingValue": 1,
        "bestRating": 5,
        "worstRating": 1,
        "alternateName": "虚假",
    },
    Verdict.MOSTLY_FALSE: {
        "ratingValue": 2,
        "bestRating": 5,
        "worstRating": 1,
        "alternateName": "大部分不实",
    },
    Verdict.MISLEADING: {
        "ratingValue": 2,
        "bestRating": 5,
        "worstRating": 1,
        "alternateName": "误导性信息",
    },
    Verdict.PARTLY_TRUE: {
        "ratingValue": 3,
        "bestRating": 5,
        "worstRating": 1,
        "alternateName": "部分属实",
    },
    Verdict.UNVERIFIABLE: {
        "ratingValue": 3,
        "bestRating": 5,
        "worstRating": 1,
        "alternateName": "无法核实",
    },
    Verdict.TRUE: {
        "ratingValue": 5,
        "bestRating": 5,
        "worstRating": 1,
        "alternateName": "属实",
    },
}


def claim_to_claimreview(cv: ClaimVerification, original_message: str = "") -> dict:
    """单条 ClaimVerification → ClaimReview JSON-LD。"""
    rating = _VERDICT_TO_RATING.get(
        cv.verdict,
        {"ratingValue": 3, "bestRating": 5, "worstRating": 1, "alternateName": "待核查"},
    )

    review_body = cv.reasoning[:500] if cv.reasoning else ""

    evidence_urls = [e.url for e in cv.evidence_chain if e.url][:5]

    return {
        "@context": "https://schema.org",
        "@type": "ClaimReview",
        "datePublished": datetime.now().isoformat(),
        "url": "",
        "claimReviewed": cv.claim.text,
        "author": {
            "@type": "Organization",
            "name": "TruthNote",
            "url": "https://github.com/AppApp777/wenke-song",
        },
        "reviewRating": {
            "@type": "Rating",
            **rating,
        },
        "itemReviewed": {
            "@type": "Claim",
            "text": cv.claim.text,
            "appearance": {
                "@type": "CreativeWork",
                "text": original_message[:200] if original_message else "",
            },
        },
        "reviewBody": review_body,
        "citation": [{"@type": "WebPage", "url": u} for u in evidence_urls],
    }


def response_to_claimreviews(response: VerifyResponse) -> list[dict]:
    """VerifyResponse → ClaimReview JSON-LD 列表。"""
    return [claim_to_claimreview(cv, response.original_message) for cv in response.claims]
