
from typing import Dict
from io import BytesIO
import requests, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_IDS = [
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov2-base"
]

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_proc = None
_loaded_id = None

def _load():
    global _model, _proc, _loaded_id
    if _model is not None:
        return
    last_err = None
    for mid in MODEL_IDS:
        try:
            _proc = AutoImageProcessor.from_pretrained(mid)
            _model = AutoModel.from_pretrained(mid)
            _model = _model.to(_device).eval()
            _loaded_id = mid
            return
        except Exception as e:
            last_err = e
            _model = None
            _proc = None
            continue
    raise RuntimeError(f"Could not load any DINO model. Last error: {last_err}")

def _get_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

@torch.inference_mode()
def embed_image(url: str) -> Dict[str, torch.Tensor]:
    _load()
    img = _get_image(url)
    inputs = _proc(images=img, return_tensors="pt").to(_device)
    out = _model(**inputs)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        cls = out.pooler_output
    else:
        cls = out.last_hidden_state[:, 0]
    return {
        "model_id": _loaded_id,
        "cls": cls.detach().cpu(),
        "tokens": out.last_hidden_state.detach().cpu()
    }
