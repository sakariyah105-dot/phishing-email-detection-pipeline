"""
relationship_memory.py

The piece classic word-based phishing detection is missing: a memory of who
normally emails whom, and a way to flag requests that don't fit that history
-- even when the email's wording is perfectly clean.

Why this matters (2026 context): most phishing today is AI-written and no
longer contains obvious suspicious language. A word-scoring model has
nothing left to catch. But an attacker impersonating "your CFO" for the
first time ever, asking for a wire transfer, is something a relationship
history can catch regardless of how well-written the email is -- because
the RELATIONSHIP itself is new or the REQUEST TYPE is new, not the words.

Storage: SQLite, a single local file, no server or setup required.

Usage:
    from relationship_memory import RelationshipMemory

    rm = RelationshipMemory("relationships.db")
    rm.record_email("vendor@example.com", "you@company.com", "payment")
    score, flags = rm.score_email("vendor@example.com", "you@company.com", "payment")
"""

import sqlite3
import datetime
from dataclasses import dataclass, field


@dataclass
class RelationshipResult:
    risk_score: int                 # 0 (fits history perfectly) - 100 (total anomaly)
    flags: list = field(default_factory=list)
    is_first_contact: bool = False
    prior_email_count: int = 0
    first_contact_date: str = None
    has_requested_this_type_before: bool = False


SENSITIVE_TYPES = {"payment", "credential", "urgent"}


class RelationshipMemory:
    def __init__(self, db_path="relationships.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                recipient TEXT NOT NULL,
                request_type TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pair
            ON emails (sender, recipient)
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Writing history
    # ------------------------------------------------------------------
    def record_email(self, sender: str, recipient: str, request_type: str,
                      timestamp: str = None):
        """Log that this email happened. Call this AFTER scoring it, once
        you've decided it's legitimate (or at least not blocked outright) --
        you don't want to teach the system that a confirmed phishing email
        is 'normal history' for that sender."""
        sender, recipient = sender.strip().lower(), recipient.strip().lower()
        timestamp = timestamp or datetime.datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO emails (sender, recipient, request_type, timestamp) VALUES (?, ?, ?, ?)",
            (sender, recipient, request_type, timestamp)
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Reading history
    # ------------------------------------------------------------------
    def _history(self, sender: str, recipient: str):
        sender, recipient = sender.strip().lower(), recipient.strip().lower()
        rows = self.conn.execute(
            "SELECT request_type, timestamp FROM emails "
            "WHERE sender=? AND recipient=? ORDER BY timestamp ASC",
            (sender, recipient)
        ).fetchall()
        return rows

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score_email(self, sender: str, recipient: str, request_type: str) -> RelationshipResult:
        """
        Score how well a new email fits this sender/recipient's established
        pattern. Pure history-based logic -- no NLP, no ML, no API calls.
        This is deliberately simple and fast: it's meant to run on every
        email, cheaply, before anything more expensive (content model, LLM)
        gets involved.
        """
        history = self._history(sender, recipient)
        flags = []

        if not history:
            score = 45
            flags.append("First-ever contact between this sender and recipient on record.")
            if request_type in SENSITIVE_TYPES:
                score += 35
                flags.append(
                    f"A '{request_type}' request arriving on the very first contact is a "
                    "classic pretexting pattern -- legitimate relationships rarely open this way."
                )
            score = min(100, score)
            return RelationshipResult(
                risk_score=score, flags=flags, is_first_contact=True,
                prior_email_count=0, first_contact_date=None,
                has_requested_this_type_before=False
            )

        prior_types = [r[0] for r in history]
        first_contact_date = history[0][1]
        prior_count = len(history)
        has_this_type_before = request_type in prior_types

        score = 5  # established relationship, small baseline
        flags.append(f"{prior_count} prior exchange(s) on record since {first_contact_date[:10]}.")

        if request_type in SENSITIVE_TYPES and not has_this_type_before:
            score += 40
            flags.append(
                f"This sender has never made a '{request_type}' request before -- "
                "this is a new kind of ask for an otherwise-established relationship."
            )
        elif request_type in SENSITIVE_TYPES and has_this_type_before:
            score += 5
            flags.append(f"This sender has made '{request_type}' requests before.")

        # Very sparse history + sensitive request = still somewhat risky,
        # even if technically "not first contact"
        if prior_count <= 2 and request_type in SENSITIVE_TYPES:
            score += 10
            flags.append("Relationship history is thin (2 or fewer prior emails) -- limited basis for trust.")

        score = min(100, score)
        return RelationshipResult(
            risk_score=score, flags=flags, is_first_contact=False,
            prior_email_count=prior_count, first_contact_date=first_contact_date,
            has_requested_this_type_before=has_this_type_before
        )

    def close(self):
        self.conn.close()


# -------------------------------------------------------------------------
# Demo / manual test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    demo_db = "demo_relationships.db"
    if os.path.exists(demo_db):
        os.remove(demo_db)  # fresh demo each run

    rm = RelationshipMemory(demo_db)

    print("=== Scenario 1: First-ever contact, asking for a payment ===")
    result = rm.score_email("new-vendor@totally-legit.com", "you@company.com", "payment")
    print(f"Risk score: {result.risk_score}")
    for f in result.flags:
        print(f"  - {f}")

    print("\n=== Scenario 2: Same sender, now with 3 months of normal history ===")
    rm.record_email("coworker@company.com", "you@company.com", "normal", "2025-01-10T09:00:00")
    rm.record_email("coworker@company.com", "you@company.com", "normal", "2025-02-14T09:00:00")
    rm.record_email("coworker@company.com", "you@company.com", "normal", "2025-03-20T09:00:00")
    result = rm.score_email("coworker@company.com", "you@company.com", "normal")
    print(f"Risk score: {result.risk_score}")
    for f in result.flags:
        print(f"  - {f}")

    print("\n=== Scenario 3: Same trusted coworker, but SUDDENLY asks for credentials ===")
    result = rm.score_email("coworker@company.com", "you@company.com", "credential")
    print(f"Risk score: {result.risk_score}")
    for f in result.flags:
        print(f"  - {f}")
    print("\n  -> This is the case a word-based model would likely MISS if the")
    print("     email is well-written -- the anomaly is behavioral, not lexical.")

    rm.close()
