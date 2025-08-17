import re
from difflib import SequenceMatcher

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

def needs_year(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["when","what year","year","date","built","founded","founded in","established"])

def needs_location(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["where","which city","which country","located","location"])

def trivial_no_retrieval(question: str) -> bool:
    q = question.lower()
    # skip retrieval for easy category questions
    return any(k in q for k in ["what animal","what color","how many","what sport"])

def has_year(text: str) -> bool:
    return YEAR_RE.search(text or "") is not None

def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def answer_supported_by(title: str, summary: str, answer: str) -> bool:
    # require the answer head token or a close title match
    toks = [t for t in answer.split() if t.isalpha()]
    head = toks[0].lower() if toks else ""
    in_text = head and (head in (summary or "").lower() or head in (title or "").lower())
    title_close = title_similarity(answer, title) >= 0.6 if title else False
    return in_text or title_close
