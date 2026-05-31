import tempfile

import pytest

from src.truthnote.memory import (
    MemoryStore,
    _claim_fingerprint,
    _entities_compatible,
    _extract_entities,
)
from src.truthnote.schemas import (
    Claim,
    ClaimVerification,
    Evidence,
    RumorCategory,
    Verdict,
    VerifyResponse,
)


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield MemoryStore(db_path=f"{tmpdir}/test_memory.db")


def _make_response(message: str = "测试消息", verdict: Verdict = Verdict.FALSE) -> VerifyResponse:
    claim = Claim(text="存款超5万要交税", category=RumorCategory.POLICY)
    evidence = Evidence(source="人民网", snippet="经核实不实", url="https://example.com")
    cv = ClaimVerification(
        claim=claim,
        verdict=verdict,
        confidence=0.9,
        evidence_chain=[evidence],
        reasoning="官方辟谣",
    )
    return VerifyResponse(
        original_message=message,
        claims=[cv],
        overall_verdict=verdict,
        summary="经核查为谣言",
        friendly_reply="爸，这个消息不太准确哦",
    )


def test_save_and_recall_claim_exact(store: MemoryStore):
    resp = _make_response()
    case_id = store.save_case(resp)
    assert case_id == 1

    candidates = store.recall_claim_candidates("存款超5万要交税")
    assert len(candidates) >= 1
    assert candidates[0]["verdict"] == "谣言"
    assert candidates[0]["match_type"] == "exact"


def test_recall_claim_fingerprint_same_order(store: MemoryStore):
    resp = _make_response()
    store.save_case(resp)
    candidates = store.recall_claim_candidates("存款超5万要交税")
    assert len(candidates) >= 1


def test_recall_claim_fingerprint_different_order_no_match(store: MemoryStore):
    resp = _make_response()
    store.save_case(resp)
    candidates = store.recall_claim_candidates("交税超存款5万要")
    exact_or_fp = [c for c in candidates if c["match_type"] in ("exact", "fingerprint")]
    assert len(exact_or_fp) == 0


def test_recall_claim_miss(store: MemoryStore):
    candidates = store.recall_claim_candidates("完全不相关的内容")
    assert len(candidates) == 0


def test_recall_case_exact(store: MemoryStore):
    resp = _make_response(message="紧急通知存款超5万要交税")
    store.save_case(resp)
    case = store.recall_case_exact("紧急通知存款超5万要交税")
    assert case is not None
    assert case["overall_verdict"] == "谣言"


def test_recall_case_exact_miss(store: MemoryStore):
    resp = _make_response(message="紧急通知存款超5万要交税")
    store.save_case(resp)
    case = store.recall_case_exact("完全不同的消息")
    assert case is None


def test_restore_claim_verification(store: MemoryStore):
    resp = _make_response()
    store.save_case(resp)
    cv = store.restore_claim_verification(1)
    assert cv is not None
    assert cv.verdict == Verdict.FALSE
    assert cv.claim.text == "存款超5万要交税"


def test_restore_claim_verification_missing(store: MemoryStore):
    cv = store.restore_claim_verification(999)
    assert cv is None


def test_source_credibility(store: MemoryStore):
    assert store.get_source_credibility("gov.cn") == "S"
    assert store.get_source_credibility("people.com.cn") == "A"
    assert store.get_source_credibility("random.xyz") == "未评级"


def test_save_feedback(store: MemoryStore):
    resp = _make_response()
    case_id = store.save_case(resp)
    store.save_feedback(case_id, "更温和", "希望语气再软一点")
    stats = store.get_stats()
    assert stats["total_feedback"] == 1


def test_stats(store: MemoryStore):
    resp = _make_response()
    store.save_case(resp)
    stats = store.get_stats()
    assert stats["total_cases"] == 1
    assert stats["total_claims"] == 1
    assert stats["total_evidence"] == 1
    assert stats["total_memories"] == 1
    conn = store._conn()
    count = conn.execute("SELECT COUNT(*) FROM source_registry").fetchone()[0]
    conn.close()
    assert stats["total_sources"] == count


def test_bump_hit_count(store: MemoryStore):
    resp = _make_response()
    store.save_case(resp)
    candidates = store.recall_claim_candidates("存款超5万要交税")
    assert len(candidates) >= 1
    memory_id = candidates[0]["id"]
    store.bump_hit_count(memory_id)
    store.bump_hit_count(memory_id)
    stats = store.get_stats()
    assert stats["top_hits"][0]["hit_count"] == 2


def test_has_negative_feedback(store: MemoryStore):
    resp = _make_response()
    case_id = store.save_case(resp)
    assert store.has_negative_feedback(case_id) is False
    store.save_feedback(case_id, "incorrect", "判定有误")
    assert store.has_negative_feedback(case_id) is True


def test_fingerprint_consistency():
    assert _claim_fingerprint("存款超5万") == _claim_fingerprint("存款超5万")
    assert _claim_fingerprint("AB CD") != _claim_fingerprint("CD AB")
    assert _claim_fingerprint("A给B转账") != _claim_fingerprint("B给A转账")


def test_extract_entities_numbers():
    ents = _extract_entities("存款超5万要交税")
    assert "5万" in ents


def test_extract_entities_percentage():
    ents = _extract_entities("人民币贬值30%")
    assert "30%" in ents


def test_extract_entities_date():
    ents = _extract_entities("2026年3月起执行")
    assert any("2026" in e for e in ents)


def test_extract_entities_relative_time():
    ents = _extract_entities("下个月开始收费")
    assert "下个月" in ents


def test_entities_compatible_same_number():
    assert _entities_compatible("存款超5万要交税", "存款超5万交个人所得税") is True


def test_entities_incompatible_different_number():
    assert _entities_compatible("存款超5万要交税", "存款超10万要交税") is False


def test_entities_compatible_no_entities():
    assert _entities_compatible("柠檬水能治癌", "柠檬水可以杀死癌细胞") is True


def test_entities_compatible_one_side_empty():
    assert _entities_compatible("存款超5万要交税", "银行存款要交税") is True
