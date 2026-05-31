"""TruthNote 记忆层。

SQLite 持久化，6 张表：
- cases: 核查案例（一次完整核查）
- claims: 声明记录
- evidence: 证据记录
- memory: 记忆库（精确/指纹/FTS 三种匹配）
- source_registry: 来源权威分级
- feedback: 用户反馈

核心能力：
1. 相同谣言换说法命中 → 秒回（路演杀手锏）
2. 来源可信度查询 → 证据加权
3. 用户反馈 → 持续改进
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import time
from pathlib import Path

from .schemas import ClaimVerification, VerifyResponse

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/truthnote_memory.db"

SOURCE_CREDIBILITY_SEEDS = [
    ("gov.cn", "S", "政府官网"),
    ("piyao.org.cn", "S", "中国互联网联合辟谣平台"),
    ("xinhuanet.com", "A", "新华社"),
    ("people.com.cn", "A", "人民网"),
    ("cctv.com", "A", "央视"),
    ("nhc.gov.cn", "S", "国家卫健委"),
    ("pbc.gov.cn", "S", "中国人民银行"),
    ("csrc.gov.cn", "S", "证监会"),
    ("moj.gov.cn", "S", "司法部"),
    ("12321.cn", "A", "网络不良信息举报中心"),
    ("who.int", "A", "世界卫生组织"),
    ("reuters.com", "A", "路透社"),
    ("ap.org", "A", "美联社"),
    ("nature.com", "A", "Nature"),
    ("weibo.com", "C", "微博"),
    ("zhihu.com", "C", "知乎"),
    ("toutiao.com", "C", "今日头条"),
    ("mp.weixin.qq.com", "C", "微信公众号"),
    ("douyin.com", "D", "抖音"),
    ("kuaishou.com", "D", "快手"),
]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_embedding(text: str) -> list[float]:
    try:
        from . import llm

        return llm.get_embedding(text)
    except Exception:
        return []


def _extract_entities(text: str) -> set[str]:
    """提取关键实体：数字+单位、百分比、日期，用于语义匹配时防误召回。"""
    entities: set[str] = set()
    for m in re.finditer(r"\d+[\.\d]*\s*[万亿元块岁%％度天月年倍杯克斤公里米级]", text):
        entities.add(re.sub(r"\s+", "", m.group()))
    for m in re.finditer(r"\d{4}[-/年]\d{1,2}[-/月]?(?:\d{1,2}[日号]?)?", text):
        entities.add(m.group())
    for m in re.finditer(r"(?:下个?月|明[天年]|今[天年晚]|上个?月|去年|前天)", text):
        entities.add(m.group())
    return entities


def _entities_compatible(query_text: str, stored_text: str) -> bool:
    """比对两段文本的关键实体，有冲突则不兼容。"""
    q_ents = _extract_entities(query_text)
    s_ents = _extract_entities(stored_text)
    if not q_ents and not s_ents:
        return True
    if not q_ents or not s_ents:
        return True
    q_nums = {e for e in q_ents if re.match(r"\d", e)}
    s_nums = {e for e in s_ents if re.match(r"\d", e)}
    if q_nums and s_nums and q_nums != s_nums:
        return False
    q_time = {e for e in q_ents if not re.match(r"\d", e)}
    s_time = {e for e in s_ents if not re.match(r"\d", e)}
    return not (q_time and s_time and q_time != s_time)


def _claim_fingerprint(text: str) -> str:
    """生成声明指纹：保留词序的规范化哈希。

    保留中文标点（逗号句号等影响语义），只去除空格和 emoji。
    防止"不，收费"和"不收费"碰撞。
    """
    clean = re.sub(r"[\s​-‏️]", "", text)
    clean = re.sub(
        r"[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f900-\U0001f9ff]",
        "",
        clean,
    )
    return hashlib.sha256(clean.encode()).hexdigest()


class MemoryStore:
    """SQLite 记忆层。"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else Path(DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_message TEXT NOT NULL,
                overall_verdict TEXT NOT NULL,
                summary TEXT,
                friendly_reply TEXT,
                response_json TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL REFERENCES cases(id),
                text TEXT NOT NULL,
                category TEXT,
                verdict TEXT,
                confidence REAL,
                reasoning TEXT,
                fingerprint TEXT
            );

            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL REFERENCES claims(id),
                source TEXT,
                url TEXT,
                title TEXT,
                snippet TEXT,
                credibility TEXT,
                supports_claim INTEGER
            );

            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                verdict TEXT NOT NULL,
                summary TEXT,
                friendly_reply TEXT,
                case_id INTEGER REFERENCES cases(id),
                claim_id INTEGER REFERENCES claims(id),
                hit_count INTEGER DEFAULT 0,
                embedding_json TEXT,
                invalidated_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_registry (
                domain TEXT PRIMARY KEY,
                credibility_grade TEXT NOT NULL,
                description TEXT
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER REFERENCES cases(id),
                feedback_type TEXT NOT NULL,
                content TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lifecycle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL REFERENCES cases(id),
                current_state TEXT NOT NULL DEFAULT 'detected',
                risk_level TEXT NOT NULL DEFAULT 'medium',
                intervention_type TEXT DEFAULT '',
                self_correction_sent INTEGER DEFAULT 0,
                escalation_deadline REAL,
                escalation_message TEXT DEFAULT '',
                self_correction_script TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lifecycle_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lifecycle_id INTEGER NOT NULL REFERENCES lifecycle(id),
                from_state TEXT NOT NULL,
                to_state TEXT NOT NULL,
                detail TEXT DEFAULT '',
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_fingerprint ON memory(fingerprint);
            CREATE INDEX IF NOT EXISTS idx_claims_fingerprint ON claims(fingerprint);
            CREATE INDEX IF NOT EXISTS idx_claims_case_id ON claims(case_id);
            CREATE INDEX IF NOT EXISTS idx_evidence_claim_id ON evidence(claim_id);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_case_id ON lifecycle(case_id);
            """
        )
        # --- 自动迁移：旧数据库可能缺少新列 ---
        try:
            conn.execute("SELECT embedding_json FROM memory LIMIT 0")
        except Exception:
            conn.execute("ALTER TABLE memory ADD COLUMN embedding_json TEXT")
        conn.commit()

        for col, typ in [("invalidated_at", "REAL"), ("claim_id", "INTEGER")]:
            try:
                conn.execute(f"ALTER TABLE memory ADD COLUMN {col} {typ}")
                conn.commit()
            except sqlite3.OperationalError:
                pass

        for col, typ in [
            ("escalation_deadline", "REAL"),
            ("escalation_message", "TEXT DEFAULT ''"),
            ("self_correction_script", "TEXT DEFAULT ''"),
        ]:
            try:
                conn.execute(f"ALTER TABLE lifecycle ADD COLUMN {col} {typ}")
                conn.commit()
            except sqlite3.OperationalError:
                pass

        existing = conn.execute("SELECT COUNT(*) FROM source_registry").fetchone()[0]
        if existing == 0:
            conn.executemany(
                "INSERT OR IGNORE INTO source_registry"
                " (domain, credibility_grade, description)"
                " VALUES (?, ?, ?)",
                SOURCE_CREDIBILITY_SEEDS,
            )
            conn.commit()
            logger.info("[Memory] 初始化来源权威分级：%d 条", len(SOURCE_CREDIBILITY_SEEDS))

        conn.close()

    def save_case(self, response: VerifyResponse) -> int:
        """保存一次完整核查结果。"""
        conn = self._conn()
        now = time.time()

        cur = conn.execute(
            "INSERT INTO cases"
            " (original_message, overall_verdict, summary,"
            " friendly_reply, response_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                response.original_message,
                response.overall_verdict.value,
                response.summary,
                response.friendly_reply,
                response.model_dump_json(),
                now,
            ),
        )
        case_id = cur.lastrowid

        for cv in response.claims:
            fp = _claim_fingerprint(cv.claim.text)
            cur2 = conn.execute(
                "INSERT INTO claims"
                " (case_id, text, category, verdict, confidence, reasoning, fingerprint)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    cv.claim.text,
                    cv.claim.category.value,
                    cv.verdict.value,
                    cv.confidence,
                    cv.reasoning,
                    fp,
                ),
            )
            claim_id = cur2.lastrowid

            for ev in cv.evidence_chain:
                conn.execute(
                    "INSERT INTO evidence"
                    " (claim_id, source, url, title, snippet, credibility, supports_claim)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        claim_id,
                        ev.source,
                        ev.url,
                        ev.title,
                        ev.snippet,
                        ev.credibility,
                        1 if ev.supports_claim else 0 if ev.supports_claim is not None else None,
                    ),
                )

            embedding = _get_embedding(cv.claim.text)
            embedding_json_str = json.dumps(embedding) if embedding else None
            conn.execute(
                "INSERT OR REPLACE INTO memory"
                " (text, fingerprint, verdict, summary, friendly_reply,"
                " case_id, claim_id, hit_count, embedding_json,"
                " created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
                (
                    cv.claim.text,
                    fp,
                    cv.verdict.value,
                    response.summary,
                    response.friendly_reply,
                    case_id,
                    claim_id,
                    embedding_json_str,
                    now,
                    now,
                ),
            )

        conn.commit()
        conn.close()
        logger.info("[Memory] 保存案例 #%d：%d 条声明", case_id, len(response.claims))
        return case_id

    def recall_case_exact(self, original_message: str) -> dict | None:
        """案例级精确召回：只匹配完整消息原文。"""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM cases WHERE original_message = ? ORDER BY created_at DESC LIMIT 1",
            (original_message,),
        ).fetchone()
        conn.close()
        if row:
            logger.info("[Memory] 案例精确命中：%s", original_message[:30])
            return dict(row)
        return None

    def recall_claim_candidates(self, claim_text: str) -> list[dict]:
        """声明级候选召回：返回所有可能匹配的记忆行，不做复用决策。

        调用方根据 match_type + compatibility 决定是否复用。
        排除已失效（invalidated_at 非空）的记忆条目。
        负反馈同时按 case_id 和 fingerprint 检查（防止换消息绕过）。
        """
        conn = self._conn()
        candidates: list[dict] = []
        fp = _claim_fingerprint(claim_text)

        def _check_neg(row_dict: dict) -> bool:
            if row_dict.get("case_id") and self.has_negative_feedback(row_dict["case_id"]):
                return True
            return bool(
                row_dict.get("fingerprint")
                and self.has_negative_feedback_by_fingerprint(row_dict["fingerprint"])
            )

        # 精确匹配
        row = conn.execute(
            "SELECT * FROM memory WHERE text = ? AND invalidated_at IS NULL"
            " ORDER BY updated_at DESC LIMIT 1",
            (claim_text,),
        ).fetchone()
        if row:
            r = dict(row)
            r["match_type"] = "exact"
            r["similarity"] = 1.0
            r["reusable"] = not _check_neg(r)
            candidates.append(r)
            conn.close()
            return candidates

        # 指纹匹配
        row = conn.execute(
            "SELECT * FROM memory WHERE fingerprint = ? AND invalidated_at IS NULL"
            " ORDER BY updated_at DESC LIMIT 1",
            (fp,),
        ).fetchone()
        if row:
            compatible = _entities_compatible(claim_text, row["text"])
            r = dict(row)
            r["match_type"] = "fingerprint"
            r["similarity"] = 0.95
            r["reusable"] = compatible and not _check_neg(r)
            candidates.append(r)
            if r["reusable"]:
                conn.close()
                return candidates

        # 语义匹配
        query_embedding = _get_embedding(claim_text)
        if query_embedding:
            rows = conn.execute(
                "SELECT * FROM memory WHERE embedding_json IS NOT NULL"
                " AND invalidated_at IS NULL"
                " ORDER BY updated_at DESC LIMIT 100"
            ).fetchall()
            for mem_row in rows:
                stored_emb = json.loads(mem_row["embedding_json"])
                sim = _cosine_similarity(query_embedding, stored_emb)
                if sim >= 0.82:
                    compatible = _entities_compatible(claim_text, mem_row["text"])
                    r = dict(mem_row)
                    r["match_type"] = "semantic"
                    r["similarity"] = round(sim, 3)
                    r["reusable"] = compatible and not _check_neg(r)
                    candidates.append(r)

        conn.close()
        return candidates

    def restore_claim_verification(self, claim_id: int) -> ClaimVerification | None:
        """通过 claim_id 精确恢复单条声明的核查结果。"""
        conn = self._conn()
        cr = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        if not cr:
            conn.close()
            return None

        ev_rows = conn.execute(
            "SELECT * FROM evidence WHERE claim_id = ? ORDER BY id",
            (claim_id,),
        ).fetchall()
        conn.close()

        from .schemas import Claim, Evidence, RumorCategory, Verdict

        try:
            verdict = Verdict(cr["verdict"])
        except (ValueError, KeyError):
            logger.warning(
                "[Memory] 无效的存储 verdict（claim_id=%s）：%r", claim_id, cr["verdict"]
            )
            return None

        evidence_list = [
            Evidence(
                source=er["source"] or "",
                url=er["url"] or "",
                title=er["title"] or "",
                snippet=er["snippet"] or "",
                credibility=er["credibility"] or "未评估",
                supports_claim=bool(er["supports_claim"])
                if er["supports_claim"] is not None
                else None,
            )
            for er in ev_rows
        ]

        category = cr["category"] or "其他"
        try:
            cat_enum = RumorCategory(category)
        except ValueError:
            cat_enum = RumorCategory.OTHER

        return ClaimVerification(
            claim=Claim(text=cr["text"], category=cat_enum),
            verdict=verdict,
            confidence=cr["confidence"] or 0.5,
            evidence_chain=evidence_list,
            reasoning=cr["reasoning"] or "",
        )

    def bump_hit_count(self, memory_id: int) -> None:
        """增加记忆命中计数。"""
        conn = self._conn()
        conn.execute(
            "UPDATE memory SET hit_count = hit_count + 1, updated_at = ? WHERE id = ?",
            (time.time(), memory_id),
        )
        conn.commit()
        conn.close()

    def get_full_case(self, case_id: int) -> dict | None:
        """从 cases/claims/evidence 还原完整核查数据（供记忆命中时返回）。"""
        conn = self._conn()
        case_row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if not case_row:
            conn.close()
            return None

        # 如果有完整 JSON，直接返回
        if case_row["response_json"]:
            try:
                result = json.loads(case_row["response_json"])
                conn.close()
                return result
            except (json.JSONDecodeError, TypeError):
                logger.warning("[Memory] case_id=%d 的 response_json 损坏，从表结构还原", case_id)

        # 否则从表结构还原
        claim_rows = conn.execute(
            "SELECT * FROM claims WHERE case_id = ? ORDER BY id", (case_id,)
        ).fetchall()
        claims_data = []
        evidence_sources = []
        for cr in claim_rows:
            ev_rows = conn.execute(
                "SELECT * FROM evidence WHERE claim_id = ? ORDER BY id", (cr["id"],)
            ).fetchall()
            evidence_list = []
            for er in ev_rows:
                evidence_list.append(
                    {
                        "source": er["source"],
                        "url": er["url"] or "",
                        "title": er["title"] or "",
                        "snippet": er["snippet"] or "",
                        "credibility": er["credibility"] or "未评估",
                        "supports_claim": bool(er["supports_claim"])
                        if er["supports_claim"] is not None
                        else None,
                    }
                )
                if er["url"] and er["url"] not in evidence_sources:
                    evidence_sources.append(er["url"])

            claims_data.append(
                {
                    "claim": {
                        "text": cr["text"],
                        "category": cr["category"] or "其他",
                        "original_context": "",
                    },
                    "verdict": cr["verdict"],
                    "confidence": cr["confidence"] or 0.5,
                    "evidence_chain": evidence_list,
                    "reasoning": cr["reasoning"] or "",
                }
            )

        conn.close()
        return {
            "original_message": case_row["original_message"],
            "claims": claims_data,
            "overall_verdict": case_row["overall_verdict"],
            "summary": case_row["summary"] or "",
            "friendly_reply": case_row["friendly_reply"] or "",
            "evidence_sources": evidence_sources,
        }

    def has_negative_feedback(self, case_id: int) -> bool:
        """检查某案例是否有负面反馈（用户纠错）。"""
        conn = self._conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE case_id = ?"
            " AND feedback_type IN ("
            "'incorrect','disagree','不对','有误','判错了','错误','wrong')",
            (case_id,),
        ).fetchone()[0]
        conn.close()
        return count > 0

    def has_negative_feedback_by_fingerprint(self, fingerprint: str) -> bool:
        """按 fingerprint 查负反馈——同一 claim 换消息重发时也能拦住。"""
        conn = self._conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM feedback f"
            " JOIN claims c ON f.case_id = c.case_id"
            " WHERE c.fingerprint = ?"
            " AND f.feedback_type IN"
            " ('incorrect', 'disagree', '不对', '有误', '判错了', '错误', 'wrong')",
            (fingerprint,),
        ).fetchone()[0]
        conn.close()
        return count > 0

    def invalidate_memory_by_case(self, case_id: int) -> int:
        """失效某案例关联的所有记忆条目。返回受影响行数。"""
        conn = self._conn()
        now = time.time()
        cur = conn.execute(
            "UPDATE memory SET invalidated_at = ?, updated_at = ?"
            " WHERE case_id = ? AND invalidated_at IS NULL",
            (now, now, case_id),
        )
        affected = cur.rowcount
        conn.commit()
        conn.close()
        if affected:
            logger.info("[Memory] 失效 case_id=%d 的 %d 条记忆", case_id, affected)
        return affected

    def update_case_verdict(self, case_id: int, new_verdict: str, new_summary: str) -> None:
        """回写案例判定（recheck 用）。"""
        conn = self._conn()
        conn.execute(
            "UPDATE cases SET overall_verdict = ?, summary = ? WHERE id = ?",
            (new_verdict, new_summary, case_id),
        )
        conn.commit()
        conn.close()

    def get_source_credibility(self, domain: str) -> str:
        """查来源权威等级。"""
        conn = self._conn()
        for d in [domain, ".".join(domain.split(".")[-2:])]:
            row = conn.execute(
                "SELECT credibility_grade FROM source_registry WHERE domain = ?", (d,)
            ).fetchone()
            if row:
                conn.close()
                return row["credibility_grade"]
        conn.close()
        return "未评级"

    def save_feedback(self, case_id: int, feedback_type: str, content: str = "") -> None:
        """保存用户反馈。"""
        conn = self._conn()
        conn.execute(
            "INSERT INTO feedback (case_id, feedback_type, content, created_at)"
            " VALUES (?, ?, ?, ?)",
            (case_id, feedback_type, content, time.time()),
        )
        conn.commit()
        conn.close()

    def get_case_by_id(self, case_id: int) -> dict | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT id, original_message, overall_verdict, summary,"
            " friendly_reply, response_json, created_at FROM cases WHERE id = ?",
            (case_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_cases(self, limit: int = 50) -> list[dict]:
        """返回最近核查案例列表。"""
        conn = self._conn()
        rows = conn.execute(
            "SELECT id, original_message, overall_verdict, summary,"
            " friendly_reply, created_at FROM cases"
            " ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        cases = []
        for r in rows:
            case = dict(r)
            feedbacks = conn.execute(
                "SELECT feedback_type FROM feedback WHERE case_id = ?", (r["id"],)
            ).fetchall()
            case["dispositions"] = [f["feedback_type"] for f in feedbacks]
            case["claim_count"] = conn.execute(
                "SELECT COUNT(*) FROM claims WHERE case_id = ?", (r["id"],)
            ).fetchone()[0]
            cases.append(case)
        conn.close()
        return cases

    def get_stats(self) -> dict:
        """返回记忆库统计。"""
        conn = self._conn()
        stats = {
            "total_cases": conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0],
            "total_claims": conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0],
            "total_evidence": conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0],
            "total_memories": conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0],
            "total_sources": conn.execute("SELECT COUNT(*) FROM source_registry").fetchone()[0],
            "total_feedback": conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0],
            "top_hits": [
                dict(r)
                for r in conn.execute(
                    "SELECT text, verdict, hit_count FROM memory ORDER BY hit_count DESC LIMIT 5"
                ).fetchall()
            ],
        }
        conn.close()
        return stats

    # ── 生命周期管理 ──

    def create_lifecycle(self, case_id: int, risk_level: str = "medium") -> int:
        conn = self._conn()
        now = time.time()
        cur = conn.execute(
            "INSERT INTO lifecycle"
            " (case_id, current_state, risk_level, created_at, updated_at)"
            " VALUES (?, 'detected', ?, ?, ?)",
            (case_id, risk_level, now, now),
        )
        lifecycle_id = cur.lastrowid
        conn.commit()
        conn.close()
        return lifecycle_id

    def advance_lifecycle(self, case_id: int, new_state: str, detail: str = "") -> None:
        conn = self._conn()
        now = time.time()
        row = conn.execute(
            "SELECT id, current_state FROM lifecycle WHERE case_id = ?"
            " ORDER BY created_at DESC LIMIT 1",
            (case_id,),
        ).fetchone()
        if not row:
            conn.close()
            return
        conn.execute(
            "INSERT INTO lifecycle_transitions"
            " (lifecycle_id, from_state, to_state, detail, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (row["id"], row["current_state"], new_state, detail, now),
        )
        conn.execute(
            "UPDATE lifecycle SET current_state = ?, updated_at = ? WHERE id = ?",
            (new_state, now, row["id"]),
        )
        conn.commit()
        conn.close()

    def update_lifecycle_intervention(self, case_id: int, intervention_type: str) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE lifecycle SET intervention_type = ?, updated_at = ?"
            " WHERE id = ("
            "   SELECT id FROM lifecycle WHERE case_id = ?"
            "   ORDER BY created_at DESC LIMIT 1"
            " )",
            (intervention_type, time.time(), case_id),
        )
        conn.commit()
        conn.close()

    def get_lifecycle(self, case_id: int) -> dict | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM lifecycle WHERE case_id = ? ORDER BY created_at DESC LIMIT 1",
            (case_id,),
        ).fetchone()
        if not row:
            conn.close()
            return None
        transitions = conn.execute(
            "SELECT from_state, to_state, detail, created_at"
            " FROM lifecycle_transitions WHERE lifecycle_id = ?"
            " ORDER BY created_at",
            (row["id"],),
        ).fetchall()
        conn.close()
        return {
            **dict(row),
            "transitions": [dict(t) for t in transitions],
        }

    def get_all_lifecycles(self, limit: int = 50) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT l.*, c.original_message, c.overall_verdict"
            " FROM lifecycle l JOIN cases c ON l.case_id = c.id"
            " ORDER BY l.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            transitions = conn.execute(
                "SELECT from_state, to_state, detail, created_at"
                " FROM lifecycle_transitions WHERE lifecycle_id = ?"
                " ORDER BY created_at",
                (r["id"],),
            ).fetchall()
            d["transitions"] = [dict(t) for t in transitions]
            result.append(d)
        conn.close()
        return result

    def start_tracking(
        self,
        case_id: int,
        escalation_deadline: float,
        escalation_message: str,
        self_correction_script: str,
    ) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE lifecycle SET escalation_deadline = ?, escalation_message = ?,"
            " self_correction_script = ?, updated_at = ?"
            " WHERE case_id = ? AND id = ("
            "   SELECT id FROM lifecycle WHERE case_id = ?"
            "   ORDER BY created_at DESC LIMIT 1"
            " )",
            (
                escalation_deadline,
                escalation_message,
                self_correction_script,
                time.time(),
                case_id,
                case_id,
            ),
        )
        conn.commit()
        conn.close()

    def mark_self_corrected(self, case_id: int) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE lifecycle SET self_correction_sent = 1, updated_at = ?"
            " WHERE id = ("
            "   SELECT id FROM lifecycle WHERE case_id = ?"
            "   ORDER BY created_at DESC LIMIT 1"
            " )",
            (time.time(), case_id),
        )
        conn.commit()
        conn.close()

    def get_pending_escalations(self) -> list[dict]:
        conn = self._conn()
        now = time.time()
        rows = conn.execute(
            "SELECT l.*, c.original_message, c.overall_verdict"
            " FROM lifecycle l JOIN cases c ON l.case_id = c.id"
            " WHERE l.current_state = 'tracking'"
            "   AND l.escalation_deadline IS NOT NULL"
            "   AND l.escalation_deadline <= ?"
            "   AND l.self_correction_sent = 0"
            " ORDER BY l.escalation_deadline",
            (now,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_tracking_cases(self) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT l.*, c.original_message, c.overall_verdict"
            " FROM lifecycle l JOIN cases c ON l.case_id = c.id"
            " WHERE l.current_state IN ('tracking', 'intervened')"
            " ORDER BY l.created_at DESC",
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
