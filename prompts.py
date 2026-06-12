"""
Prompt contract for the ICU-transfer clinical-agent benchmark.

One stateless, single-turn call per patient. The model must choose exactly one of
three actions and return a strict, parseable JSON object so the decision can be
scored deterministically.

    A = Continue routine floor monitoring
    B = Order routine morning labs
    C = Escalate to ICU

The output contract is intentionally rigid (single-line JSON, action as a bare
letter) so that parse-success is a first-class, measurable signal alongside the
clinical decision itself.
"""

from __future__ import annotations
import json
import re

ACTIONS = {
    "A": "Continue routine floor monitoring",
    "B": "Order routine morning labs",
    "C": "Escalate to ICU",
}

SYSTEM_PROMPT = (
    "You are the attending physician covering a general medical/surgical ward. "
    "A nurse pages you about one patient. You are given ONLY what has been observed "
    "in the last 12 hours on the floor (vital signs and laboratory results). "
    "Nothing after this moment is known to you. Decide the single best next action.\n\n"
    "Choose EXACTLY ONE option:\n"
    "  A = Continue routine floor monitoring\n"
    "  B = Order routine morning labs\n"
    "  C = Escalate to ICU (rapid response / ICU transfer)\n\n"
    "Reason like a clinician weighing deterioration risk: trends and trajectory matter "
    "more than any single value; consider tachycardia, hypotension, tachypnea, hypoxia, "
    "fever/hypothermia, rising lactate, worsening renal function, leukocytosis, and "
    "acid-base derangement. Escalate (C) when the trajectory suggests impending "
    "instability; do not escalate stable patients.\n\n"
    "Return ONLY a single-line JSON object, no prose before or after, in exactly this form:\n"
    '{"action": "A", "rationale": "<= 30 words"}\n'
    "The value of \"action\" MUST be the single capital letter A, B, or C."
)

USER_TEMPLATE = (
    "PATIENT SNAPSHOT (general ward, last 12 hours; decision time is now):\n\n"
    "{narrative}\n\n"
    "Question: What is the single best next action for this patient right now? "
    'Respond with ONLY the JSON object: {{"action": "A|B|C", "rationale": "..."}}'
)


def build_user_prompt(narrative: str) -> str:
    return USER_TEMPLATE.format(narrative=narrative)


_LETTER_RE = re.compile(r"\b([ABC])\b")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_action(value):
    """Map a model's 'action' value to A/B/C WITHOUT truncating (so the word
    'Continue' is never read as 'C'). Returns the letter or None."""
    v = str(value).strip().upper()
    if v in ACTIONS:                       # already a bare letter
        return v
    if len(v) >= 2 and v[0] in ACTIONS and not v[1].isalnum():
        return v[0]                        # "A)", "A.", "A -"
    if v.startswith("CONTINUE") or "FLOOR MONITOR" in v:
        return "A"
    if v.startswith("ORDER") or "MORNING LAB" in v:
        return "B"
    if v.startswith("ESCALATE") or "ICU" in v:
        return "C"
    return None


def parse_action(raw_text: str):
    """
    Robustly extract the chosen action letter and rationale from a model response.

    Returns (action, rationale, parse_ok).
    parse_ok is True only when a clean A/B/C decision could be recovered.
    Strategy: try strict JSON first, then a loose JSON search, then a bare-letter
    fallback. The fallback still counts as parse_ok=False to keep format adherence
    honest as its own metric.
    """
    if raw_text is None:
        return None, "", False
    text = raw_text.strip()

    # 1) strict JSON
    try:
        obj = json.loads(text)
        act = _coerce_action(obj.get("action", ""))
        if act in ACTIONS:
            return act, str(obj.get("rationale", ""))[:300], True
    except Exception:
        pass

    # 2) first {...} block in the text
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            act = _coerce_action(obj.get("action", ""))
            if act in ACTIONS:
                return act, str(obj.get("rationale", ""))[:300], True
        except Exception:
            pass

    # 3) bare-letter fallback (parse_ok = False: format contract was not honored)
    # Prefer an explicit "action": X or "ANSWER: X" pattern if present.
    m2 = re.search(r'action["\s:]+([ABC])', text, re.IGNORECASE)
    if not m2:
        m2 = re.search(r'answer[\s:]+([ABC])', text, re.IGNORECASE)
    if not m2:
        m2 = _LETTER_RE.search(text)
    if m2:
        return m2.group(1).upper(), text[:300], False

    return None, text[:300], False
