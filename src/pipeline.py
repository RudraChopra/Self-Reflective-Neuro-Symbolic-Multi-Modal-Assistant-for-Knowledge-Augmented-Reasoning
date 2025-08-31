
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from io import BytesIO
import requests, torch, re, wikipedia, os

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None

def _load_blip():
    global _model, _processor
    if _model is None:
        model_id = "Salesforce/blip-vqa-base"
        _processor = BlipProcessor.from_pretrained(model_id)
        _model = BlipForQuestionAnswering.from_pretrained(model_id).to(device).eval()

def _load_image_any(src: str) -> Image.Image:
    # URL case
    if isinstance(src, str) and src.lower().startswith(("http://","https://")):
        r = requests.get(src, headers={"User-Agent":"ColabRunner"}, timeout=30)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    # Local path case
    p = os.path.abspath(src)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Image not found at {p}")
    return Image.open(p).convert("RGB")

def vqa(image_src: str, question: str) -> str:
    _load_blip()
    img = _load_image_any(image_src)
    inputs = _processor(img, question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=20)
    return _processor.decode(out[0], skip_special_tokens=True)

def retrieve_wiki_best(query: str, hint: str = "", k: int = 5, sentences: int = 3):
    try:
        candidates = wikipedia.search(query + " " + hint if hint else query, results=k)
        best_title, best_sum, best_score = "", "", 0.0
        for t in candidates or []:
            try:
                page = wikipedia.page(t, auto_suggest=False, redirect=True)
                score = 1.0 if hint and hint.lower() in page.title.lower() else 0.0
                if score > best_score:
                    best_title, best_sum, best_score = page.title, wikipedia.summary(page.title, sentences=sentences), score
            except Exception:
                continue
        return best_title, best_sum
    except Exception:
        return "", ""

def reflect(question: str, answer: str, title: str, summary: str) -> str:
    import re
    needs_year = any(k in question.lower() for k in ["when","year","date","built","founded","established"])
    has_year = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", answer or "") is not None
    if needs_year and not has_year:
        return "needs more info"
    if summary and any(w in (summary.lower()+" "+title.lower()) for w in (answer or "").lower().split()[:2]):
        return "confident"
    return "confident" if summary else "needs more info"

def qa_with_retrieval(image_src: str, question: str):
    ans1 = vqa(image_src, question)
    title, ev = retrieve_wiki_best(question, hint=ans1, k=5, sentences=3)
    status = reflect(question, ans1, title, ev)
    if status != "confident":
        ans2 = vqa(image_src, f"{question}. Use facts about {title}")
        return {"answer": ans2, "evidence_title": title, "evidence": ev, "reflection": status}
    return {"answer": ans1, "evidence_title": title, "evidence": ev, "reflection": status}
