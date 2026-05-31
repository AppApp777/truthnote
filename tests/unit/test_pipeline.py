from src.truthnote.pipeline import verify_message
from src.truthnote.schemas import Verdict


def test_verify_message_full_pipeline(mock_llm_pipeline):
    result = verify_message("紧急通知！银行存款超过5万元要交20%的税！赶紧取钱！", use_memory=False)

    assert result.original_message.startswith("紧急通知")
    assert len(result.claims) == 1
    assert result.claims[0].verdict == Verdict.FALSE
    assert result.claims[0].confidence > 0.8
    assert result.overall_verdict == Verdict.FALSE
    # 强辟谣规则命中时跳过 FactChecker+Skeptic，ResponseComposer mock 响应可能错位
    assert mock_llm_pipeline["n"] >= 7


def test_verify_empty_claims(mock_llm_empty):
    result = verify_message("今天天气真好啊")
    assert result.overall_verdict == Verdict.UNVERIFIABLE
    assert len(result.claims) == 0
