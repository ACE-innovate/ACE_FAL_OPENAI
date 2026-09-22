"""ACE_FAL_OPENAI: GPT Image 2.5 nodes (fal.ai) - Flare & Sunburst, Text-to-Image & Edit.

Drop into ComfyUI custom_nodes/ (single file) and restart.
API key: node input or FAL_KEY environment variable.

Endpoints (schemas per fal llms.txt, 2026-09; flare and sunburst share one API surface):
  - openai/gpt-image-2.5/flare/text-to-image
  - openai/gpt-image-2.5/flare/edit
  - openai/gpt-image-2.5/sunburst/text-to-image
  - openai/gpt-image-2.5/sunburst/edit"""ACE_FAL_OPENAI: GPT Image 2.5 nodes (fal.ai) - Flare & Sunburst, Text-to-Image & Edit.

Drop into ComfyUI custom_nodes/ (single file) and restart.
API key: node input or FAL_KEY environment variable.

Endpoints (schemas per fal llms.txt, 2026-09; flare and sunburst share one API surface):
  - openai/gpt-image-2.5/flare/text-to-image
  - openai/gpt-image-2.5/flare/edit
  - openai/gpt-image-2.5/sunburst/text-to-image
  - openai/gpt-image-2.5/sunburst/edit
"""

import base64
import json
import os
import time
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image

FAL_BASE = "https://fal.run/openai/gpt-image-2.5"
TIMEOUT = 600

SIZE_PRESETS = [
    "auto",
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
    "custom",
]
QUALITY_OPTIONS = ["auto", "low", "medium", "high", "xhigh", "max"]
BACKGROUND_OPTIONS = ["auto", "transparent", "opaque"]
FORMAT_OPTIONS = ["png", "jpeg", "webp"]

SIZE_TOOLTIP = (
    "Preset, 'auto', or 'custom' (uses custom_width/height). Custom: multiples of 16, "
    "max edge 3840, aspect ratio <= 3:1, total pixels 655,360..8,294,400."
)
QUALITY_TOOLTIP = (
    "Higher = more detail, latency and COST (token-billed; default high). "
    "'auto' lets the model choose."
)


def _output_dir() -> str:
    try:
        import folder_paths
        d = folder_paths.get_output_directory()
    except Exception:
        d = os.path.join(os.getcwd(), "output")
    os.makedirs(d, exist_ok=True)
    return d


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    t = x.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _pil_to_tensor_rgb(pil: Image.Image) -> torch.Tensor:
    pil.load()  # force full decode before buffer reuse
    arr = np.array(pil.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr))[None, ...]


def _placeholder(size: int = 512) -> torch.Tensor:
    return _pil_to_tensor_rgb(Image.new("RGB", (size, size), (100, 100, 100)))


def _tensor_to_data_uri(x: torch.Tensor) -> str:
    buf = BytesIO()
    _tensor_to_pil(x).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _tensor_frames_to_data_uris(x: torch.Tensor) -> List[str]:
    t = x.detach().cpu()
    if t.ndim == 3:
        t = t[None, ...]
    return [_tensor_to_data_uri(t[i]) for i in range(t.shape[0])]


def _mask_to_data_uri(mask: torch.Tensor) -> str:
    """MASK [B,H,W] -> RGBA PNG data URI. Convention (OpenAI edit): TRANSPARENT
    pixels mark the region to edit; here mask=1 (white) becomes transparent =
    editable. If results look inverted, invert the mask upstream."""
    m = mask.detach().cpu()
    if m.ndim == 3:
        m = m[0]
    a = ((1.0 - m.clamp(0, 1)).numpy() * 255).astype(np.uint8)  # 1 -> alpha 0
    h, w = a.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = a
    buf = BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _get_key(api_key: str) -> str:
    key = (api_key or "").strip() or os.getenv("FAL_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ACE_FAL_OPENAI ERROR: No API key in node input or FAL_KEY environment variable."
        )
    return key


def _fal_post(url: str, key: str, payload: dict) -> dict:
    resp = requests.post(
        url,
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"fal API {resp.status_code}: {resp.text[:2000]}")
    return resp.json()


def _download(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def _save_raw(raw: bytes, name: str, log: List[str]) -> Optional[str]:
    try:
        p = os.path.join(_output_dir(), name)
        with open(p, "wb") as f:
            f.write(raw)
        log.append(f"raw saved -> {p}")
        return p
    except Exception as e:
        log.append(f"raw save failed ({name}): {e}")
        return None


def _stack_rgb(pils: List[Image.Image], log: Optional[List[str]] = None) -> torch.Tensor:
    if not pils:
        return _placeholder()
    w = min(p.width for p in pils)
    h = min(p.height for p in pils)
    out = []
    for p in pils:
        if p.size != (w, h):
            p = p.resize((w, h), Image.LANCZOS)
        out.append(_pil_to_tensor_rgb(p))
    t = torch.cat(out, 0).contiguous().to(torch.float32).clamp(0.0, 1.0)
    # ComfyUI IMAGE contract: [B, H, W, 3] float32 in 0..1
    if t.ndim != 4 or t.shape[-1] != 3:
        raise RuntimeError(f"IMAGE tensor contract violated: shape {tuple(t.shape)}")
    if log is not None:
        log.append(
            f"tensor out: shape {tuple(t.shape)} dtype {t.dtype} contiguous {t.is_contiguous()}"
        )
    return t


def _resolve_size(image_size: str, custom_width: int, custom_height: int, log: List[str]):
    if image_size == "custom":
        w = max(16, (int(custom_width) // 16) * 16)
        h = max(16, (int(custom_height) // 16) * 16)
        if (w, h) != (int(custom_width), int(custom_height)):
            log.append(f"custom size snapped to multiples of 16: {w}x{h}")
        return {"width": w, "height": h}
    return image_size


def _common_payload(prompt, size_val, background, quality, num_images):
    return {
        "prompt": prompt,
        "image_size": size_val,
        "background": background,
        "quality": quality,
        "num_images": int(num_images),
        "output_format": "png",  # always fetch lossless master; local conversion covers jpg/webp
    }


def _run_generation(endpoint: str, key: str, payload: dict, save_raw: bool, tag: str,
                    out_png: bool = True, out_jpg: bool = False, out_webp: bool = False,
                    jpg_webp_quality: int = 90):
    log: List[str] = []
    t0 = time.time()
    data = _fal_post(endpoint, key, payload)
    log.append(f"API call completed in {time.time() - t0:.1f}s")

    images = data.get("images") or []
    if not images:
        raise RuntimeError(f"No images returned. Response: {json.dumps(data)[:1500]}")

    if not (out_png or out_jpg or out_webp):
        out_png = True
        log.append("no format ticked; defaulting to png")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    pils: List[Image.Image] = []
    raw_paths: List[str] = []
    for idx, meta in enumerate(images):
        url = meta.get("url")
        if not url:
            continue
        raw = _download(url)
        pil = Image.open(BytesIO(raw))
        pil.load()
        pils.append(pil)
        if save_raw:
            base = f"gpt25_{tag}_{stamp}_{idx + 1:02d}"
            if out_png:
                rp = _save_raw(raw, f"{base}.png", log)  # untouched API bytes
                if rp:
                    raw_paths.append(rp)
            try:
                if out_jpg:
                    p = os.path.join(_output_dir(), f"{base}.jpg")
                    pil.convert("RGB").save(p, "JPEG", quality=int(jpg_webp_quality))
                    raw_paths.append(p)
                    log.append(f"saved -> {p}")
                if out_webp:
                    p = os.path.join(_output_dir(), f"{base}.webp")
                    pil.save(p, "WEBP", quality=int(jpg_webp_quality))
                    raw_paths.append(p)
                    log.append(f"saved -> {p}")
            except Exception as e:
                log.append(f"format conversion failed: {e}")

    log.append(f"{len(pils)} image(s) generated")
    t = _stack_rgb(pils, log)
    return t, "\n".join(raw_paths), "\n".join(log)


# =====================================================================
# Base classes (shared layouts); endpoint set by subclass
# =====================================================================

class _AceGPT25T2IBase:
    _ENDPOINT = ""
    _TAG = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "password": True, "tooltip": "fal API key (or set FAL_KEY env)"},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (SIZE_PRESETS, {"default": "landscape_4_3", "tooltip": SIZE_TOOLTIP}),
                "quality": (QUALITY_OPTIONS, {"default": "high", "tooltip": QUALITY_TOOLTIP}),
            },
            "optional": {
                "background": (BACKGROUND_OPTIONS, {"default": "auto"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "Not sent to the API (endpoint has no seed parameter). Forces re-execution: identical inputs are otherwise served from ComfyUI's cache with no new API call."}),
                "out_png": ("BOOLEAN", {"default": True, "tooltip": "Save results as .png"}),
                "out_jpg": ("BOOLEAN", {"default": False, "tooltip": "Also save results as .jpg"}),
                "out_webp": ("BOOLEAN", {"default": False, "tooltip": "Also save results as .webp"}),
                "jpg_webp_quality": ("INT", {"default": 90, "min": 1, "max": 100, "tooltip": "Quality for jpg/webp saves"}),
                "custom_width": ("INT", {"default": 1024, "min": 480, "max": 3840, "step": 16}),
                "custom_height": ("INT", {"default": 1024, "min": 480, "max": 3840, "step": 16}),
                "save_raw": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Save untouched result files to the output folder"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "raw_paths", "operation_log")
    FUNCTION = "run"
    CATEGORY = "ACE_FAL_OPENAI"

    def run(
        self,
        api_key: str,
        prompt: str,
        image_size: str = "landscape_4_3",
        quality: str = "high",
        background: str = "auto",
        num_images: int = 1,
        seed: int = 0,  # cache-buster only; endpoint has no seed parameter
        out_png: bool = True,
        out_jpg: bool = False,
        out_webp: bool = False,
        jpg_webp_quality: int = 90,
        custom_width: int = 1024,
        custom_height: int = 1024,
        save_raw: bool = True,
        **kwargs,
    ):
        key = _get_key(api_key)
        if not prompt.strip():
            raise RuntimeError("Prompt is required.")
        log: List[str] = []
        size_val = _resolve_size(image_size, custom_width, custom_height, log)
        payload = _common_payload(prompt.strip(), size_val, background, quality, num_images)
        images, raw_paths, run_log = _run_generation(
            self._ENDPOINT, key, payload, save_raw, self._TAG,
            out_png, out_jpg, out_webp, jpg_webp_quality,
        )
        return (images, raw_paths, ("\n".join(log) + "\n" + run_log).strip())


class _AceGPT25EditBase:
    _ENDPOINT = ""
    _TAG = ""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "mask": (
                "MASK",
                {
                    "forceInput": False,
                    "tooltip": "Optional edit region. White (1) = editable area (sent as transparency per OpenAI convention). Invert upstream if results are flipped.",
                },
            ),
            "background": (BACKGROUND_OPTIONS, {"default": "auto"}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "Not sent to the API (endpoint has no seed parameter). Forces re-execution: identical inputs are otherwise served from ComfyUI's cache with no new API call."}),
            "out_png": ("BOOLEAN", {"default": True, "tooltip": "Save results as .png"}),
            "out_jpg": ("BOOLEAN", {"default": False, "tooltip": "Also save results as .jpg"}),
            "out_webp": ("BOOLEAN", {"default": False, "tooltip": "Also save results as .webp"}),
            "jpg_webp_quality": ("INT", {"default": 90, "min": 1, "max": 100, "tooltip": "Quality for jpg/webp saves"}),
            "custom_width": ("INT", {"default": 1024, "min": 480, "max": 3840, "step": 16}),
            "custom_height": ("INT", {"default": 1024, "min": 480, "max": 3840, "step": 16}),
            "save_raw": (
                "BOOLEAN",
                {"default": True, "tooltip": "Save untouched result files to the output folder"},
            ),
            "images_batch": (
                "IMAGE",
                {
                    "forceInput": False,
                    "tooltip": "Batch input; every frame is sent as a separate reference image.",
                },
            ),
        }
        for i in range(1, 9):
            opt[f"image_{i}"] = ("IMAGE", {"forceInput": False})
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "password": True, "tooltip": "fal API key (or set FAL_KEY env)"},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (SIZE_PRESETS, {"default": "auto", "tooltip": SIZE_TOOLTIP + " 'auto' infers from inputs."}),
                "quality": (QUALITY_OPTIONS, {"default": "high", "tooltip": QUALITY_TOOLTIP}),
            },
            "optional": opt,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "raw_paths", "operation_log")
    FUNCTION = "run"
    CATEGORY = "ACE_FAL_OPENAI"

    def run(
        self,
        api_key: str,
        prompt: str,
        image_size: str = "auto",
        quality: str = "high",
        background: str = "auto",
        num_images: int = 1,
        seed: int = 0,  # cache-buster only; endpoint has no seed parameter
        out_png: bool = True,
        out_jpg: bool = False,
        out_webp: bool = False,
        jpg_webp_quality: int = 90,
        custom_width: int = 1024,
        custom_height: int = 1024,
        save_raw: bool = True,
        **kwargs,
    ):
        key = _get_key(api_key)
        if not prompt.strip():
            raise RuntimeError("Prompt is required.")
        log: List[str] = []

        image_urls: List[str] = []
        batch = kwargs.get("images_batch")
        if isinstance(batch, torch.Tensor):
            image_urls.extend(_tensor_frames_to_data_uris(batch))
        for i in range(1, 9):
            im = kwargs.get(f"image_{i}")
            if isinstance(im, torch.Tensor):
                image_urls.extend(_tensor_frames_to_data_uris(im))
        if not image_urls:
            raise RuntimeError("Connect at least one reference image (image_1..image_8 or images_batch).")
        if len(image_urls) > 16:
            log.append(f"{len(image_urls)} reference images; API allows 16 - trimming to first 16.")
            image_urls = image_urls[:16]

        size_val = _resolve_size(image_size, custom_width, custom_height, log)
        payload = _common_payload(prompt.strip(), size_val, background, quality, num_images)
        payload["image_urls"] = image_urls

        mask = kwargs.get("mask")
        if isinstance(mask, torch.Tensor):
            payload["mask_url"] = _mask_to_data_uri(mask)
            log.append("mask attached (white = editable, sent as transparency)")

        images, raw_paths, run_log = _run_generation(
            self._ENDPOINT, key, payload, save_raw, self._TAG,
            out_png, out_jpg, out_webp, jpg_webp_quality,
        )
        return (images, raw_paths, ("\n".join(log) + "\n" + run_log).strip())


# =====================================================================
# The four nodes
# =====================================================================

class AceGPT25FlareT2I(_AceGPT25T2IBase):
    """Flare: OpenAI's default model - fast, high quality, everyday generation."""
    _ENDPOINT = f"{FAL_BASE}/flare/text-to-image"
    _TAG = "flare_t2i"
    DESCRIPTION = "GPT Image 2.5 Flare text-to-image via fal: fast default-quality generation, transparent backgrounds supported."


class AceGPT25FlareEdit(_AceGPT25EditBase):
    """Flare: precise editing, fast variant."""
    _ENDPOINT = f"{FAL_BASE}/flare/edit"
    _TAG = "flare_edit"
    DESCRIPTION = "GPT Image 2.5 Flare edit via fal: prompt + up to 16 reference images, optional mask."


class AceGPT25SunburstT2I(_AceGPT25T2IBase):
    """Sunburst: precision model - intricate detail, longer generation times."""
    _ENDPOINT = f"{FAL_BASE}/sunburst/text-to-image"
    _TAG = "sunburst_t2i"
    DESCRIPTION = "GPT Image 2.5 Sunburst text-to-image via fal: maximum-fidelity generation for intricate detail."


class AceGPT25SunburstEdit(_AceGPT25EditBase):
    """Sunburst: tightest-control editing, composition preserved across revisions."""
    _ENDPOINT = f"{FAL_BASE}/sunburst/edit"
    _TAG = "sunburst_edit"
    DESCRIPTION = "GPT Image 2.5 Sunburst edit via fal: precision edits, up to 16 reference images, optional mask."


NODE_CLASS_MAPPINGS = {
    "AceGPT25FlareT2I": AceGPT25FlareT2I,
    "AceGPT25FlareEdit": AceGPT25FlareEdit,
    "AceGPT25SunburstT2I": AceGPT25SunburstT2I,
    "AceGPT25SunburstEdit": AceGPT25SunburstEdit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AceGPT25FlareT2I": "ACE GPT-2.5 Flare Text-to-Image (fal)",
    "AceGPT25FlareEdit": "ACE GPT-2.5 Flare Edit (fal)",
    "AceGPT25SunburstT2I": "ACE GPT-2.5 Sunburst Text-to-Image (fal)",
    "AceGPT25SunburstEdit": "ACE GPT-2.5 Sunburst Edit (fal)",
}
"""

import base64
import json
import os
import time
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image

FAL_BASE = "https://fal.run/openai/gpt-image-2.5"
TIMEOUT = 600

SIZE_PRESETS = [
    "auto",
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
    "custom",
]
QUALITY_OPTIONS = ["auto", "low", "medium", "high", "xhigh", "max"]
BACKGROUND_OPTIONS = ["auto", "transparent", "opaque"]
FORMAT_OPTIONS = ["png", "jpeg", "webp"]

SIZE_TOOLTIP = (
    "Preset, 'auto', or 'custom' (uses custom_width/height). Custom: multiples of 16, "
    "max edge 3840, aspect ratio <= 3:1, total pixels 655,360..8,294,400."
)
QUALITY_TOOLTIP = (
    "Higher = more detail, latency and COST (token-billed; default high). "
    "'auto' lets the model choose."
)


def _output_dir() -> str:
    try:
        import folder_paths
        d = folder_paths.get_output_directory()
    except Exception:
        d = os.path.join(os.getcwd(), "output")
    os.makedirs(d, exist_ok=True)
    return d


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    t = x.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _pil_to_tensor_rgb(pil: Image.Image) -> torch.Tensor:
    pil.load()  # force full decode before buffer reuse
    arr = np.array(pil.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr))[None, ...]


def _placeholder(size: int = 512) -> torch.Tensor:
    return _pil_to_tensor_rgb(Image.new("RGB", (size, size), (100, 100, 100)))


def _tensor_to_data_uri(x: torch.Tensor) -> str:
    buf = BytesIO()
    _tensor_to_pil(x).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _tensor_frames_to_data_uris(x: torch.Tensor) -> List[str]:
    t = x.detach().cpu()
    if t.ndim == 3:
        t = t[None, ...]
    return [_tensor_to_data_uri(t[i]) for i in range(t.shape[0])]


def _mask_to_data_uri(mask: torch.Tensor) -> str:
    """MASK [B,H,W] -> RGBA PNG data URI. Convention (OpenAI edit): TRANSPARENT
    pixels mark the region to edit; here mask=1 (white) becomes transparent =
    editable. If results look inverted, invert the mask upstream."""
    m = mask.detach().cpu()
    if m.ndim == 3:
        m = m[0]
    a = ((1.0 - m.clamp(0, 1)).numpy() * 255).astype(np.uint8)  # 1 -> alpha 0
    h, w = a.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = a
    buf = BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _get_key(api_key: str) -> str:
    key = (api_key or "").strip() or os.getenv("FAL_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ACE_FAL_OPENAI ERROR: No API key in node input or FAL_KEY environment variable."
        )
    return key


def _fal_post(url: str, key: str, payload: dict) -> dict:
    resp = requests.post(
        url,
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"fal API {resp.status_code}: {resp.text[:2000]}")
    return resp.json()


def _download(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def _save_raw(raw: bytes, name: str, log: List[str]) -> Optional[str]:
    try:
        p = os.path.join(_output_dir(), name)
        with open(p, "wb") as f:
            f.write(raw)
        log.append(f"raw saved -> {p}")
        return p
    except Exception as e:
        log.append(f"raw save failed ({name}): {e}")
        return None


def _stack_rgb(pils: List[Image.Image]) -> torch.Tensor:
    if not pils:
        return _placeholder()
    w = min(p.width for p in pils)
    h = min(p.height for p in pils)
    out = []
    for p in pils:
        if p.size != (w, h):
            p = p.resize((w, h), Image.LANCZOS)
        out.append(_pil_to_tensor_rgb(p))
    return torch.cat(out, 0)


def _resolve_size(image_size: str, custom_width: int, custom_height: int, log: List[str]):
    if image_size == "custom":
        w = max(16, (int(custom_width) // 16) * 16)
        h = max(16, (int(custom_height) // 16) * 16)
        if (w, h) != (int(custom_width), int(custom_height)):
            log.append(f"custom size snapped to multiples of 16: {w}x{h}")
        return {"width": w, "height": h}
    return image_size


def _common_payload(prompt, size_val, background, quality, num_images):
    return {
        "prompt": prompt,
        "image_size": size_val,
        "background": background,
        "quality": quality,
        "num_images": int(num_images),
        "output_format": "png",  # always fetch lossless master; local conversion covers jpg/webp
    }


def _run_generation(endpoint: str, key: str, payload: dict, save_raw: bool, tag: str,
                    out_png: bool = True, out_jpg: bool = False, out_webp: bool = False,
                    jpg_webp_quality: int = 90):
    log: List[str] = []
    t0 = time.time()
    data = _fal_post(endpoint, key, payload)
    log.append(f"API call completed in {time.time() - t0:.1f}s")

    images = data.get("images") or []
    if not images:
        raise RuntimeError(f"No images returned. Response: {json.dumps(data)[:1500]}")

    if not (out_png or out_jpg or out_webp):
        out_png = True
        log.append("no format ticked; defaulting to png")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    pils: List[Image.Image] = []
    raw_paths: List[str] = []
    for idx, meta in enumerate(images):
        url = meta.get("url")
        if not url:
            continue
        raw = _download(url)
        pil = Image.open(BytesIO(raw))
        pil.load()
        pils.append(pil)
        if save_raw:
            base = f"gpt25_{tag}_{stamp}_{idx + 1:02d}"
            if out_png:
                rp = _save_raw(raw, f"{base}.png", log)  # untouched API bytes
                if rp:
                    raw_paths.append(rp)
            try:
                if out_jpg:
                    p = os.path.join(_output_dir(), f"{base}.jpg")
                    pil.convert("RGB").save(p, "JPEG", quality=int(jpg_webp_quality))
                    raw_paths.append(p)
                    log.append(f"saved -> {p}")
                if out_webp:
                    p = os.path.join(_output_dir(), f"{base}.webp")
                    pil.save(p, "WEBP", quality=int(jpg_webp_quality))
                    raw_paths.append(p)
                    log.append(f"saved -> {p}")
            except Exception as e:
                log.append(f"format conversion failed: {e}")

    log.append(f"{len(pils)} image(s) generated")
    return _stack_rgb(pils), "\n".join(raw_paths), "\n".join(log)


# =====================================================================
# Base classes (shared layouts); endpoint set by subclass
# =====================================================================

class _AceGPT25T2IBase:
    _ENDPOINT = ""
    _TAG = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "password": True, "tooltip": "fal API key (or set FAL_KEY env)"},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (SIZE_PRESETS, {"default": "landscape_4_3", "tooltip": SIZE_TOOLTIP}),
                "quality": (QUALITY_OPTIONS, {"default": "high", "tooltip": QUALITY_TOOLTIP}),
            },
            "optional": {
                "background": (BACKGROUND_OPTIONS, {"default": "auto"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "out_png": ("BOOLEAN", {"default": True, "tooltip": "Save results as .png"}),
                "out_jpg": ("BOOLEAN", {"default": False, "tooltip": "Also save results as .jpg"}),
                "out_webp": ("BOOLEAN", {"default": False, "tooltip": "Also save results as .webp"}),
                "jpg_webp_quality": ("INT", {"default": 90, "min": 1, "max": 100, "tooltip": "Quality for jpg/webp saves"}),
                "custom_width": ("INT", {"default": 1024, "min": 480, "max": 3840, "step": 16}),
                "custom_height": ("INT", {"default": 1024, "min": 480, "max": 3840, "step": 16}),
                "save_raw": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Save untouched result files to the output folder"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "raw_paths", "operation_log")
    FUNCTION = "run"
    CATEGORY = "ACE_FAL_OPENAI"

    def run(
        self,
        api_key: str,
        prompt: str,
        image_size: str = "landscape_4_3",
        quality: str = "high",
        background: str = "auto",
        num_images: int = 1,
        out_png: bool = True,
        out_jpg: bool = False,
        out_webp: bool = False,
        jpg_webp_quality: int = 90,
        custom_width: int = 1024,
        custom_height: int = 1024,
        save_raw: bool = True,
        **kwargs,
    ):
        key = _get_key(api_key)
        if not prompt.strip():
            raise RuntimeError("Prompt is required.")
        log: List[str] = []
        size_val = _resolve_size(image_size, custom_width, custom_height, log)
        payload = _common_payload(prompt.strip(), size_val, background, quality, num_images)
        images, raw_paths, run_log = _run_generation(
            self._ENDPOINT, key, payload, save_raw, self._TAG,
            out_png, out_jpg, out_webp, jpg_webp_quality,
        )
        return (images, raw_paths, ("\n".join(log) + "\n" + run_log).strip())


class _AceGPT25EditBase:
    _ENDPOINT = ""
    _TAG = ""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "mask": (
                "MASK",
                {
                    "forceInput": False,
                    "tooltip": "Optional edit region. White (1) = editable area (sent as transparency per OpenAI convention). Invert upstream if results are flipped.",
                },
            ),
            "background": (BACKGROUND_OPTIONS, {"default": "auto"}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
            "out_png": ("BOOLEAN", {"default": True, "tooltip": "Save results as .png"}),
            "out_jpg": ("BOOLEAN", {"default": False, "tooltip": "Also save results as .jpg"}),
            "out_webp": ("BOOLEAN", {"default": False, "tooltip": "Also save results as .webp"}),
            "jpg_webp_quality": ("INT", {"default": 90, "min": 1, "max": 100, "tooltip": "Quality for jpg/webp saves"}),
            "custom_width": ("INT", {"default": 1024, "min": 480, "max": 3840, "step": 16}),
            "custom_height": ("INT", {"default": 1024, "min": 480, "max": 3840, "step": 16}),
            "save_raw": (
                "BOOLEAN",
                {"default": True, "tooltip": "Save untouched result files to the output folder"},
            ),
            "images_batch": (
                "IMAGE",
                {
                    "forceInput": False,
                    "tooltip": "Batch input; every frame is sent as a separate reference image.",
                },
            ),
        }
        for i in range(1, 9):
            opt[f"image_{i}"] = ("IMAGE", {"forceInput": False})
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "password": True, "tooltip": "fal API key (or set FAL_KEY env)"},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (SIZE_PRESETS, {"default": "auto", "tooltip": SIZE_TOOLTIP + " 'auto' infers from inputs."}),
                "quality": (QUALITY_OPTIONS, {"default": "high", "tooltip": QUALITY_TOOLTIP}),
            },
            "optional": opt,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "raw_paths", "operation_log")
    FUNCTION = "run"
    CATEGORY = "ACE_FAL_OPENAI"

    def run(
        self,
        api_key: str,
        prompt: str,
        image_size: str = "auto",
        quality: str = "high",
        background: str = "auto",
        num_images: int = 1,
        out_png: bool = True,
        out_jpg: bool = False,
        out_webp: bool = False,
        jpg_webp_quality: int = 90,
        custom_width: int = 1024,
        custom_height: int = 1024,
        save_raw: bool = True,
        **kwargs,
    ):
        key = _get_key(api_key)
        if not prompt.strip():
            raise RuntimeError("Prompt is required.")
        log: List[str] = []

        image_urls: List[str] = []
        batch = kwargs.get("images_batch")
        if isinstance(batch, torch.Tensor):
            image_urls.extend(_tensor_frames_to_data_uris(batch))
        for i in range(1, 9):
            im = kwargs.get(f"image_{i}")
            if isinstance(im, torch.Tensor):
                image_urls.extend(_tensor_frames_to_data_uris(im))
        if not image_urls:
            raise RuntimeError("Connect at least one reference image (image_1..image_8 or images_batch).")
        if len(image_urls) > 16:
            log.append(f"{len(image_urls)} reference images; API allows 16 - trimming to first 16.")
            image_urls = image_urls[:16]

        size_val = _resolve_size(image_size, custom_width, custom_height, log)
        payload = _common_payload(prompt.strip(), size_val, background, quality, num_images)
        payload["image_urls"] = image_urls

        mask = kwargs.get("mask")
        if isinstance(mask, torch.Tensor):
            payload["mask_url"] = _mask_to_data_uri(mask)
            log.append("mask attached (white = editable, sent as transparency)")

        images, raw_paths, run_log = _run_generation(
            self._ENDPOINT, key, payload, save_raw, self._TAG,
            out_png, out_jpg, out_webp, jpg_webp_quality,
        )
        return (images, raw_paths, ("\n".join(log) + "\n" + run_log).strip())


# =====================================================================
# The four nodes
# =====================================================================

class AceGPT25FlareT2I(_AceGPT25T2IBase):
    """Flare: OpenAI's default model - fast, high quality, everyday generation."""
    _ENDPOINT = f"{FAL_BASE}/flare/text-to-image"
    _TAG = "flare_t2i"
    DESCRIPTION = "GPT Image 2.5 Flare text-to-image via fal: fast default-quality generation, transparent backgrounds supported."


class AceGPT25FlareEdit(_AceGPT25EditBase):
    """Flare: precise editing, fast variant."""
    _ENDPOINT = f"{FAL_BASE}/flare/edit"
    _TAG = "flare_edit"
    DESCRIPTION = "GPT Image 2.5 Flare edit via fal: prompt + up to 16 reference images, optional mask."


class AceGPT25SunburstT2I(_AceGPT25T2IBase):
    """Sunburst: precision model - intricate detail, longer generation times."""
    _ENDPOINT = f"{FAL_BASE}/sunburst/text-to-image"
    _TAG = "sunburst_t2i"
    DESCRIPTION = "GPT Image 2.5 Sunburst text-to-image via fal: maximum-fidelity generation for intricate detail."


class AceGPT25SunburstEdit(_AceGPT25EditBase):
    """Sunburst: tightest-control editing, composition preserved across revisions."""
    _ENDPOINT = f"{FAL_BASE}/sunburst/edit"
    _TAG = "sunburst_edit"
    DESCRIPTION = "GPT Image 2.5 Sunburst edit via fal: precision edits, up to 16 reference images, optional mask."


NODE_CLASS_MAPPINGS = {
    "AceGPT25FlareT2I": AceGPT25FlareT2I,
    "AceGPT25FlareEdit": AceGPT25FlareEdit,
    "AceGPT25SunburstT2I": AceGPT25SunburstT2I,
    "AceGPT25SunburstEdit": AceGPT25SunburstEdit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AceGPT25FlareT2I": "ACE GPT-2.5 Flare Text-to-Image (fal)",
    "AceGPT25FlareEdit": "ACE GPT-2.5 Flare Edit (fal)",
    "AceGPT25SunburstT2I": "ACE GPT-2.5 Sunburst Text-to-Image (fal)",
    "AceGPT25SunburstEdit": "ACE GPT-2.5 Sunburst Edit (fal)",
}
