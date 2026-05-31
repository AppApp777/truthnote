"""沉默策略：4 类不该核查的输入检测。

TruthNote 不在以下 4 类上提供真假判定：
1. 政治敏感（信源对称性失败 + 用户风险转嫁 + 工具中立性）
2. 信仰民俗（可证伪性边界 / 范畴错误）
3. 未来预测（不可核实）
4. 修辞段子（字面 false 是范畴错误）

注：传统养生（中医/食疗/节气）不在沉默清单——这是可证伪的医学命题，
应走正常核查流水线；伪医疗诈骗（根治/包治）由 detect_miracle_cure 处理。

返回 dict(category, template) 或 None。命中后整条消息走 UNVERIFIABLE +
特定 template，不进入核查流水线。
"""

from src.truthnote.orchestrator import detect_silence_zone

# ── 类别 1：政治敏感 ──


def test_silence_political_figure():
    """涉及国家领导人 → 沉默。"""
    result = detect_silence_zone("听说中央政治局最近开了什么会，要调整政策")
    assert result is not None
    assert result["category"] == "political"


def test_silence_political_event():
    """群体性事件 → 沉默。"""
    result = detect_silence_zone("听说某地最近有大规模抗议活动")
    assert result is not None
    assert result["category"] == "political"


def test_silence_diplomatic_stance():
    """外交立场判断 → 沉默。"""
    result = detect_silence_zone("美国又制裁中国了，咱们是不是该硬刚")
    assert result is not None
    assert result["category"] == "political"


def test_silence_territorial_dispute():
    """领土争端 → 沉默。"""
    result = detect_silence_zone("钓鱼岛问题到底咱们能不能收回来")
    assert result is not None
    assert result["category"] == "political"


def test_political_not_triggered_by_neutral():
    """普通经济新闻不应触发政治沉默。"""
    result = detect_silence_zone("今年 GDP 增长是多少")
    assert result is None


# ── 类别 2：信仰民俗 ──


def test_silence_religion():
    """宗教教义 → 沉默。"""
    result = detect_silence_zone("佛祖说过这辈子的因果都是上辈子积的")
    assert result is not None
    assert result["category"] == "religion_folk"


def test_silence_fengshui():
    """风水问题 → 沉默。"""
    result = detect_silence_zone("祖坟朝向不对会影响后代的运势")
    assert result is not None
    assert result["category"] == "religion_folk"


def test_silence_fortune_telling():
    """算命八字 → 沉默。"""
    result = detect_silence_zone("今年犯太岁要戴红绳化煞")
    assert result is not None
    assert result["category"] == "religion_folk"


def test_silence_bazi():
    """八字推断 → 沉默。"""
    result = detect_silence_zone("八字相冲的夫妻在一起会折寿")
    assert result is not None
    assert result["category"] == "religion_folk"


def test_religion_negation_skip():
    """否定语境不触发：「别信风水那一套」。"""
    result = detect_silence_zone("别信风水那一套，都是骗人的")
    # 在否定/警告语境下不应触发沉默——这是用户在澄清，不是声明
    assert result is None or result["category"] != "religion_folk"


# ── 传统养生 NOT 沉默 ──
# 传统养生表述（祛湿/养生/上火/坐月子）应走正常核查流水线返 UNVERIFIABLE，
# 不被 silence_zone 吞掉。伪医疗诈骗信号由 detect_miracle_cure 处理。


def test_traditional_food_therapy_goes_through_pipeline():
    """传统食疗陈述应走流水线，不沉默。"""
    result = detect_silence_zone("夏天每天喝生姜水可以祛湿气")
    assert result is None


def test_traditional_yuezi_goes_through_pipeline():
    """坐月子习俗陈述应走流水线，不沉默。"""
    result = detect_silence_zone("坐月子千万不能洗头会落下病根")
    assert result is None


def test_traditional_jieqi_goes_through_pipeline():
    """节气养生陈述应走流水线，不沉默。"""
    result = detect_silence_zone("三伏天贴三伏贴可以冬病夏治")
    assert result is None


# ── 类别 3：未来预测 ──


def test_silence_prediction_economy():
    """经济预测 → 沉默。"""
    result = detect_silence_zone("明年房价肯定还会涨")
    assert result is not None
    assert result["category"] == "prediction"


def test_silence_prediction_stock():
    """股市预测 → 沉默。"""
    result = detect_silence_zone("下个月 A 股一定会暴涨")
    assert result is not None
    # 注：这条也含金融诈骗信号，但金融诈骗规则会在 orchestrator 兜底；
    # silence_zone 在前置阶段，应该让 prediction 类触发
    assert result["category"] in ("prediction", "political")


def test_silence_prediction_personal():
    """个人未来预测 → 沉默。"""
    result = detect_silence_zone("我女儿以后一定会有出息考上清华")
    assert result is not None
    assert result["category"] == "prediction"


def test_prediction_not_triggered_by_past():
    """已发生的事实陈述不触发预测沉默。"""
    result = detect_silence_zone("去年房价确实涨了百分之十")
    # 这是已发生的事实，不是预测
    assert result is None or result["category"] != "prediction"


# ── 类别 4：修辞段子 ──


def test_silence_obvious_absurd():
    """明显荒谬的反讽 → 沉默。"""
    result = detect_silence_zone("震惊！太阳从西边升起了")
    assert result is not None
    assert result["category"] == "rhetoric_joke"


def test_silence_exaggeration():
    """夸张到不可能后果 → 沉默。"""
    result = detect_silence_zone("我妈做的饭好吃得能让人撑死")
    assert result is not None
    assert result["category"] == "rhetoric_joke"


def test_silence_hypothetical():
    """假设性问题（如果...会...）→ 沉默。"""
    result = detect_silence_zone("如果地球停转 10 秒会发生什么")
    assert result is not None
    assert result["category"] == "rhetoric_joke"


# ── 通用：返回值结构 + 沉默不是空白 ──


def test_silence_returns_template():
    """命中沉默时必须返回 friendly_reply template。"""
    result = detect_silence_zone("听说中央最近开了大会")
    assert result is not None
    assert "template" in result
    assert len(result["template"]) > 20  # 模板不能空


def test_silence_template_mentions_category():
    """模板应该说清楚不判定的范畴，而非空泛的「无法核实」。"""
    result = detect_silence_zone("祖坟朝向影响后代")
    assert result is not None
    # 模板应该体现"这是宗教/民俗范畴"而非"信息不足"
    assert (
        "信仰" in result["template"]
        or "宗教" in result["template"]
        or "传统" in result["template"]
        or "范畴" in result["template"]
    )


def test_no_silence_for_normal_rumor():
    """普通谣言不应被沉默吞掉，要正常进入核查流水线。"""
    result = detect_silence_zone("紧急通知！存款超 5 万要交税")
    assert result is None
