from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from io import BytesIO
import requests, torch, wikipedia
from rules.qa_rules import needs_year, needs_location, trivial_no_retrieval, has_year, answer_supported_by, title_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None

def _load_blip():
    global _model, _processor
    if _model is None:
        model_id = "Salesforce/blip-vqa-base"
        _processor = BlipProcessor.from_pretrained(model_id)
        _model = BlipForQuestionAnswering.from_pretrained(model_id).to(device).eval()

def _load_image_bytes(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def vqa(image_url: str, question: str) -> str:
    _load_blip()
    img = _load_image_bytes(image_url)
    inputs = _processor(img, question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=20)
    return _processor.decode(out[0], skip_special_tokens=True)

def retrieve_wiki_best(query: str, hint: str = "", k: int = 5, sentences: int = 3):
    try:
        candidates = wikipedia.search(query + " " + hint if hint else query, results=k)
        best = ("", "", 0.0)
        for t in candidates or []:
            try:
                page = wikipedia.page(t, auto_suggest=False, redirect=True)
                score = max(title_similarity(hint, page.title), title_similarity(query, page.title))
                if score > best[2]:
                    best = (page.title, wikipedia.summary(page.title, sentences=sentences), score)
            except Exception:
                continue
        return best[0], best[1]
    except Exception:
        return "", ""

def reflect(question: str, answer: str, title: str, summary: str) -> str:
    if trivial_no_retrieval(question):
        return "confident"
    if needs_year(question) and not has_year(answer):
        return "needs more info"
    if needs_location(question) and not any(w in answer.lower() for w in ["city","country","state","paris","france","usa","india","china","italy","rome","london"]):
        # soft check, will be corrected by better retrieval
        pass
    if not answer_supported_by(title, summary, answer):
        return "needs more info"
    return "confident"

def qa_with_retrieval(image_url: str, question: str):
    ans1 = vqa(image_url, question)
    title, ev = ("","")
    if not trivial_no_retrieval(question):
        title, ev = retrieve_wiki_best(question, hint=ans1, k=5, sentences=3)
    status = reflect(question, ans1, title, ev)
    if status != "confident":
        title2, ev2 = retrieve_wiki_best(question, hint=ans1, k=5, sentences=5)
        if ev2:
            title, ev = title2, ev2
        ans2 = vqa(image_url, f"{question}. Use facts about {title}")
        final = ans2
        status = reflect(question, final, title, ev)
        return {"answer": final, "evidence_title": title, "evidence": ev, "reflection": status}
    return {"answer": ans1, "evidence_title": title, "evidence": ev, "reflection": status}
