# ACE_FAL_OPENAI

GPT Image 2.5 nodes for ComfyUI via [fal.ai](https://fal.ai) — Flare & Sunburst, Text-to-Image & Edit.

## Nodes

| Node | Endpoint | Purpose |
|---|---|---|
| ACE GPT-2.5 Flare Text-to-Image (fal) | `openai/gpt-image-2.5/flare/text-to-image` | Fast default-quality generation |
| ACE GPT-2.5 Flare Edit (fal) | `openai/gpt-image-2.5/flare/edit` | Fast editing, up to 16 reference images |
| ACE GPT-2.5 Sunburst Text-to-Image (fal) | `openai/gpt-image-2.5/sunburst/text-to-image` | Precision model, intricate detail, slower |
| ACE GPT-2.5 Sunburst Edit (fal) | `openai/gpt-image-2.5/sunburst/edit` | Precision editing, composition preserved |

## Install

1. Copy `ace_fal_openai.py` into `ComfyUI/custom_nodes/` (single file, no folder needed).
2. Restart ComfyUI, reload the browser.
3. Nodes appear under the **ACE_FAL_OPENAI** category.

## API key

Paste a fal key (`key_id:key_secret` format) into the node's `api_key` field, **or** set the
`FAL_KEY` environment variable before launching ComfyUI. The field wins when both exist.

## Inputs

### Common (all four nodes)
- **prompt** — required.
- **image_size** — presets (`landscape_4_3`, `portrait_16_9`, `square_hd`, ...), `auto`, or
  `custom` (uses `custom_width`/`custom_height`; snapped to multiples of 16, max edge 3840,
  aspect ratio <= 3:1). T2I defaults `landscape_4_3`; Edit defaults `auto` (inferred from inputs).
- **quality** — `auto | low | medium | high | xhigh | max`. This is the cost/latency lever
  (output is token-billed); `low`/`medium` for iteration, `high`+ for finals. Default `high`.
- **background** — `auto | transparent | opaque`. Transparent works with png/webp output.
- **num_images** — 1–10 per call.
- **seed** — NOT sent to the API (the endpoints have no seed parameter). It exists to defeat
  ComfyUI's caching: identical inputs are served from cache with no API call, which looks like
  "identical results". `randomize` forces a fresh call every queue; `fixed` deliberately reuses
  the cached result for free re-queues.
- **out_png / out_jpg / out_webp** — tick one or several. The API is always asked for a single
  lossless PNG master (one call, one cost); ticked formats are converted locally from that
  master and saved side by side. Nothing ticked falls back to png.
- **jpg_webp_quality** — 1–100 for the converted saves.
- **save_raw** — saves files to the ComfyUI output folder as
  `gpt25_<node>_<timestamp>_<n>.<ext>`. The png is the untouched API bytes.

### Edit nodes only
- **image_1..image_8** — reference images. Each socket also accepts a batch; every frame is
  sent separately.
- **images_batch** — explicit batch input (e.g. another node's multi-image output).
  Everything is merged in order and trimmed to the API maximum of 16 with a log note.
- **mask** — optional edit region. Convention (OpenAI): the TRANSPARENT area is edited; this
  node sends white (1) in the MASK as transparent = editable. If results look inverted, put an
  InvertMask node upstream.

## Outputs (all nodes)
- **images** — IMAGE batch, guaranteed `[B, H, W, 3]` float32 0..1 contiguous.
- **raw_paths** — newline-separated list of every file saved (all formats).
- **operation_log** — timing, saved paths, image count, and a final tensor-contract line
  (`tensor out: shape (...) dtype ... contiguous ...`). Wire to a Show Text node.

## Cost orientation (1024x1024, per image)
Token-billed on fal ($5/M text in, $8/M image in, $30/M image out). Output tokens by quality
(GPT Image canonical table): low ~272, medium ~1,056, high ~4,160 — i.e. roughly $0.008 /
$0.03 / $0.125 per 1024 image; xhigh/max above that. Prompt length is negligible
(100 words ~ 135 tokens). A 1024 edit request all-in at low quality ~ 6k tokens.

## Notes
- Every generated image is fetched by URL and re-downloaded immediately; fal result URLs are
  not relied on afterwards.
- `raw_paths` interoperates with the ACE C2PA nodes (`source_path` / `parent_path`) unchanged.
- Endpoints verified live 2026-09; schemas per fal's published `llms.txt` for each endpoint.
  Flare and Sunburst share one API surface; they differ in speed/detail balance, not controls.
