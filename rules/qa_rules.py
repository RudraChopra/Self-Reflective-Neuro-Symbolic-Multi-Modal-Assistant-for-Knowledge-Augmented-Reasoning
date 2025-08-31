
import re
from difflib import SequenceMatcher

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

STOP = {
    "the","a","an","of","in","on","at","by","for","to","from","and","or","is","it","this","that",
    "with","as","be","are","was","were","near","front","behind","inside","outside","harbor","harbour"
}

LOCATION_TOKENS = {
    "paris","france","london","rome","italy","new york","united states","usa","liberty island","san francisco","morocco","rabat"
}

def needs_year(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["when","what year","year","date","built","constructed","founded","established","completed","opened","inaugurated"])

def needs_location(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["where","which city","which country","located","location"])

def trivial_no_retrieval(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["what animal","what color","how many","what sport"])

def has_strict_year(text: str) -> bool:
    return YEAR_RE.search(text or "") is not None

def years_in(text: str):
    return YEAR_RE.findall(text or "")

def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def _content_tokens(text: str):
    toks = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z]+", text or "")]
    return [t for t in toks if len(t) >= 3 and t not in STOP]

def answer_supported_by(title: str, summary: str, answer: str) -> bool:
    ans_tokens = set(_content_tokens(answer))
    if not ans_tokens:
        return False
    text_tokens = set(_content_tokens((summary or "") + " " + (title or "")))
    overlap = ans_tokens & text_tokens
    if overlap:
        return True
    return bool(title) and title_similarity(answer, title) >= 0.72

def location_supported(answer: str, summary: str) -> bool:
    a = (answer or "").lower()
    s = (summary or "").lower()
    return any(tok in a and tok in s for tok in LOCATION_TOKENS)

def answer_year_supported(answer: str, summary: str) -> bool:
    yrs = years_in(answer)
    if not yrs:
        return False
    s = (summary or "").lower()
    return any(y in s for y in yrs)
