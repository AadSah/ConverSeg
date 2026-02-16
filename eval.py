#!/usr/bin/env python3
"""
eval_refcoco_sam2plm.py

Evaluate SAM2 + PLM either:
(A) on a Hugging Face dataset (default: aadarsh99/ConvSeg), or
(B) on a JSON file of items [{"id","image","mask","prompt"}, ...] via --input_json.

HF mode (default):
  - Downloads/caches the dataset automatically using `datasets.load_dataset`
  - Evaluates split(s) (default: sam_seeded,human_annotated)
  - Dataset columns expected: id, image, mask, prompt, concept
    (image/mask are paths or decodable images)

JSON mode:
  - Expects a JSON payload like:
    {
      "dataset": "chunk_01",
      "count": 723,
      "items": [
        {"id": "...", "image": "images/..png", "mask": "masks/..png", "prompt": "..."}
      ]
    }

Example (HF):
  python eval_refcoco_sam2plm.py \
    --final_ckpt ./ckpts/.../fine_tuned_sam2.torch \
    --plm_ckpt ./ckpts/.../fine_tuned_plm.torch \
    --lora_ckpt ./ckpts/.../sam2_lora.pth \
    --save_preds /tmp/convseg_preds

Example (JSON):
  python eval_refcoco_sam2plm.py \
    --input_json /path/to/items.json \
    --final_ckpt ./ckpts/.../fine_tuned_sam2.torch \
    --plm_ckpt ./ckpts/.../fine_tuned_plm.torch
"""

import os
import io
import json
import argparse
import logging
import contextlib
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import torch
import numpy as np
from tqdm.auto import tqdm

from PIL import Image, ImageFile, ImageFilter, ImageChops, ImageDraw

ImageFile.LOAD_TRUNCATED_IMAGES = True  # robust against slightly broken files

# HF datasets
try:
    from datasets import load_dataset, Image as HFImage
except Exception as e:
    load_dataset = None
    HFImage = None
    _HF_IMPORT_ERROR = e
else:
    _HF_IMPORT_ERROR = None


# --- your model bits (from your script) ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from models.language_adapter import LanguageAdapter

logger = logging.getLogger(__name__)


# ------------------------- args -------------------------

def parse_args():
    p = argparse.ArgumentParser("Evaluate SAM2+PLM on HF ConvSeg (default) or a JSON list of items")

    # ---- NEW: HF mode ----
    p.add_argument("--hf_dataset", type=str, default="aadarsh99/ConvSeg",
                   help="Hugging Face dataset repo id (default: aadarsh99/ConvSeg).")
    p.add_argument("--hf_config", type=str, default="default",
                   help="HF dataset config name (ConvSeg uses 'default').")
    p.add_argument("--hf_splits", type=str, default="sam_seeded,human_annotated",
                   help="Comma-separated splits to evaluate (e.g. 'sam_seeded' or 'sam_seeded,human_annotated').")
    p.add_argument("--hf_cache_dir", type=str, default=None,
                   help="Optional HF cache dir for datasets/downloads.")

    # ---- JSON mode ----
    p.add_argument("--input_json", type=str, default=None,
                   help="If provided, evaluate on this JSON file of {image, mask, prompt} items.")

    # ---- Model / runtime ----
    p.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml")
    p.add_argument("--base_ckpt", type=str, default="./checkpoints/sam2_hiera_large.pt")
    p.add_argument("--final_ckpt", type=str, required=True,
                   help="Your fine-tuned SAM2 checkpoint (.torch)")
    p.add_argument("--plm_ckpt", type=str, required=True,
                   help="Your PLM adapter checkpoint (.torch)")
    p.add_argument("--lora_ckpt", type=str, required=True,
                   help="LoRA adapter path")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32")

    # ---- QoL ----
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of samples (per split in HF mode; or per JSON list in JSON mode).")
    p.add_argument("--save_preds", type=str, default=None,
                   help="Directory to save predicted masks + GT + overlays.")

    return p.parse_args()


# ------------------------- model build -------------------------

def _dtype_from_precision(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _build_predictor_and_plm(cfg, base_ckpt, final_ckpt, plm_ckpt, lora_ckpt, device, precision="fp32"):
    logger.info("Building SAM2 predictor using config %s on %s", cfg, device)

    model = build_sam2(cfg, base_ckpt, device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.model.eval()

    if not os.path.isfile(final_ckpt):
        raise FileNotFoundError(f"Could not find fine-tuned SAM2 checkpoint: {final_ckpt}")
    logger.info("Loading fine-tuned SAM2 weights from %s", final_ckpt)
    sd = torch.load(final_ckpt, map_location=device)
    predictor.model.load_state_dict(sd.get("model", sd), strict=True)


    C = predictor.model.sam_mask_decoder.transformer_dim
    plm_dtype = _dtype_from_precision(precision)

    plm = LanguageAdapter(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        transformer_dim=C,
        n_sparse_tokens=0,
        use_dense_bias=True,
        use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
        dtype=plm_dtype,
        device=device,
    ).to(device)

    if not os.path.isfile(plm_ckpt):
        raise FileNotFoundError(f"Could not find PLM adapter checkpoint: {plm_ckpt}")
    logger.info("Loading PLM adapter weights from %s", plm_ckpt)
    plm_sd = torch.load(plm_ckpt, map_location=device)
    plm.load_state_dict(plm_sd["plm"], strict=True)

    if lora_ckpt is not None:
        # PLM adapter loads LoRA weights for its text/vision backbone (as in your original)
        plm.load_lora(lora_ckpt)

    return predictor, plm


class SAM2PLMWrapper(torch.nn.Module):
    """
    Thin wrapper that runs predictor + PLM and returns a single best-mask logit per prompt.
    """
    def __init__(self, predictor, plm, device="cuda", precision="fp32"):
        super().__init__()
        self.predictor = predictor
        self.plm = plm
        self.device = torch.device(device)
        self.precision = precision

    @torch.no_grad()
    def infer_one(self, rgb: np.ndarray, text: str, image_path: str) -> torch.Tensor:
        """
        Args:
            rgb: HxWx3 uint8 (RGB) in ORIGINAL RESOLUTION
            text: str
            image_path: local filepath (used by PLM adapter)
        Returns:
            logit: [H, W] logits in original image size
        """
        amp_ctx = contextlib.nullcontext()
        if self.device.type == "cuda" and self.precision in ("fp16", "bf16"):
            amp_dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)

        with amp_ctx:
            # 1) set image -> get embeddings & high-res feats
            self.predictor.set_image(rgb)
            image_emb = self.predictor._features["image_embed"][-1].unsqueeze(0)  # [1,C,Hs,Ws]
            hi = [lvl[-1].unsqueeze(0) for lvl in self.predictor._features["high_res_feats"]]

            # 2) text -> (sparse, dense) via PLM
            sp, dp = self.plm([text], image_emb.shape[-2], image_emb.shape[-1], [image_path])  # sp:[1,N,C], dp:[1,C,Hs,Ws]

            # 3) SAM2 decode
            dec = self.predictor.model.sam_mask_decoder
            dev = next(dec.parameters()).device
            dtype = next(dec.parameters()).dtype

            image_pe = self.predictor.model.sam_prompt_encoder.get_dense_pe().to(dev, dtype)
            image_emb = image_emb.to(dev, dtype)
            hi = [h.to(dev, dtype) for h in hi]
            sp, dp = sp.to(dev, dtype), dp.to(dev, dtype)

            low, scores, _, _ = dec(
                image_embeddings=image_emb,
                image_pe=image_pe,
                sparse_prompt_embeddings=sp,
                dense_prompt_embeddings=dp,
                multimask_output=True,
                repeat_image=False,
                high_res_features=hi,
            )

            # Post-process to ORIGINAL image size
            logits = self.predictor._transforms.postprocess_masks(low, self.predictor._orig_hw[-1])  # [1,3,H0,W0]
            best = scores.argmax(dim=1).item()
            logit = logits[0, best]  # [H0, W0]
            return logit


# ------------------------- basic helpers -------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _resolve_path(base_dir: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(base_dir, p)


@contextlib.contextmanager
def _suppress_stderr():
    """Last-resort silencer for noisy C-level libs (e.g., libpng via OpenCV)."""
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def _pil_read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im.info.pop("icc_profile", None)
        im = im.convert("RGB")
        return np.array(im)


def _read_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return m


def _compute_iou(pred_bool: torch.Tensor, gt_bool: torch.Tensor):
    """
    pred_bool, gt_bool: torch.bool on CPU
    """
    inter = (pred_bool & gt_bool).sum().item()
    union = (pred_bool | gt_bool).sum().item()
    iou = 1.0 if union == 0 else (inter / (union + 1e-10))
    return inter, union, iou


# ------------------------- overlay helpers ----------------------------

EDGE_COLORS_HEX = ["#FF006E"]


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


EDGE_COLORS = [_hex_to_rgb(h) for h in EDGE_COLORS_HEX]


def stable_color(key: str):
    import hashlib
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    return EDGE_COLORS[h % len(EDGE_COLORS)]


def tint(rgb, amt: float = 0.1):
    # soft pastel fill by mixing with white
    return tuple(int(255 - (255 - c) * (1 - amt)) for c in rgb)


def load_mask_bool(p: Path) -> np.ndarray:
    im = Image.open(p)
    try:
        if im.mode in ("RGBA", "LA"):
            arr = np.array(im.split()[-1]) > 0
        else:
            arr = np.array(im.convert("L")) > 0
    finally:
        im.close()
    return arr


def edge_map(mask_bool: np.ndarray, width_px: int = 1) -> Image.Image:
    m = Image.fromarray((mask_bool.astype(np.uint8) * 255), "L")
    edges = ImageChops.difference(
        m.filter(ImageFilter.MaxFilter(3)),
        m.filter(ImageFilter.MinFilter(3)),
    )
    for _ in range(max(0, width_px - 1)):
        edges = edges.filter(ImageFilter.MaxFilter(3))
    return edges.point(lambda p: 255 if p > 0 else 0)


def _resize_mask_to_target(m: np.ndarray, src_w: int, src_h: int, tgt_w: int, tgt_h: int) -> np.ndarray:
    # first resize to original image size if needed
    if (m.shape[1], m.shape[0]) != (src_w, src_h):
        m = np.array(
            Image.fromarray((m.astype(np.uint8) * 255), "L").resize((src_w, src_h), Image.NEAREST)
        ) > 0
    # then resize to target size
    m = np.array(
        Image.fromarray((m.astype(np.uint8) * 255), "L").resize((tgt_w, tgt_h), Image.NEAREST)
    ) > 0
    return m


def _apply_rounded_corners(img_rgb: Image.Image, radius: int) -> Image.Image:
    w, h = img_rgb.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    bg = Image.new("RGB", (w, h), "white")
    img_rgba = img_rgb.convert("RGBA")
    img_rgba.putalpha(mask)
    bg.paste(img_rgba.convert("RGB"), (0, 0), mask)
    return bg


def compose_overlay(
    img_path: Path,
    mask_path: Path,
    tgt_w: int,
    tgt_h: int,
    alpha_fill: float = 0.70,
    edge_w: int = 2,
    draw_box: bool = False,
    dpi: int = 300,
    no_resize: bool = False,
) -> Image.Image:
    """
    Overlay a single Boolean mask onto an image with soft pastel fill + colored contour.

    If no_resize=True, the image is kept at ORIGINAL SIZE and tgt_w/tgt_h are ignored.
    """
    base = Image.open(img_path).convert("RGB")
    src_w, src_h = base.size

    if no_resize:
        base_rgba = base.convert("RGBA")
        tgt_w, tgt_h = src_w, src_h
        m = load_mask_bool(mask_path)
        if (m.shape[1], m.shape[0]) != (src_w, src_h):
            base.close()
            raise RuntimeError(
                f"--no-resize: mask size {m.shape[1]}x{m.shape[0]} differs from image size {src_w}x{src_h}"
            )
    else:
        base_rgba = base.resize((tgt_w, tgt_h), Image.BICUBIC).convert("RGBA")
        m = load_mask_bool(mask_path)
        m = _resize_mask_to_target(m, src_w, src_h, tgt_w, tgt_h)

    color = stable_color(str(mask_path))
    fill_rgb = tint(color, 0.1)
    a = int(round(alpha_fill * 255))

    fill_layer = Image.new("RGBA", (tgt_w, tgt_h), fill_rgb + (0,))
    fill_layer.putalpha(Image.fromarray((m.astype(np.uint8) * a), "L"))
    edgesL = edge_map(m, width_px=edge_w)
    stroke = Image.new("RGBA", (tgt_w, tgt_h), color + (0,))
    stroke.putalpha(edgesL)

    out = Image.alpha_composite(base_rgba, fill_layer)
    out = Image.alpha_composite(out, stroke)

    if draw_box:
        ys, xs = np.where(m)
        if ys.size:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            pad = max(2, int(round(min(tgt_w, tgt_h) * 0.004)))
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(tgt_w - 1, x1 + pad)
            y1 = min(tgt_h - 1, y1 + pad)
            box = Image.new("RGBA", (tgt_w, tgt_h), (0, 0, 0, 0))
            d = ImageDraw.Draw(box)
            d.rectangle([x0, y0, x1, y1], outline=tint(color, 0.80) + (140,), width=2)
            out = Image.alpha_composite(out, box)

    out = out.convert("RGB")
    out = _apply_rounded_corners(out, max(12, int(0.06 * min(out.size))))
    out.info["dpi"] = (dpi, dpi)
    base.close()
    return out


# ------------------------- HF helpers -------------------------

def _mask_pil_to_u8(mask_img: Image.Image) -> np.ndarray:
    """
    Convert a PIL mask (L / 1 / RGB / RGBA) into uint8 grayscale (0..255).
    Uses alpha channel if present; otherwise converts to L.
    """
    if mask_img.mode in ("RGBA", "LA"):
        alpha = mask_img.split()[-1]
        return np.array(alpha, dtype=np.uint8)
    return np.array(mask_img.convert("L"), dtype=np.uint8)


def _hf_field_to_pil_and_path(
    field,
    tmp_dir: str,
    stem: str,
    force_rgb: bool = False,
) -> Tuple[Image.Image, str]:
    """
    HF datasets may return:
      - PIL.Image.Image
      - dict with keys like {"path": ..., "bytes": ...}
      - string path (common in JSON datasets)
    Return (PIL image, local_file_path). If no reliable path exists, write a temp PNG.
    """
    # String path
    if isinstance(field, str):
        if not os.path.isfile(field):
            # In some datasets, string paths are relative to cwd; try as Path
            if os.path.isfile(os.path.abspath(field)):
                field = os.path.abspath(field)
            else:
                raise FileNotFoundError(f"HF field path does not exist: {field}")
        img = Image.open(field)
        img = img.convert("RGB") if force_rgb else img
        return img, field

    # PIL
    if isinstance(field, Image.Image):
        img = field.convert("RGB") if force_rgb else field
        path = getattr(field, "filename", None)
        if path and os.path.isfile(path):
            return img, path
        out_path = os.path.join(tmp_dir, f"{stem}.png")
        img.save(out_path)
        return img, out_path

    # dict
    if isinstance(field, dict):
        p = field.get("path", None)
        b = field.get("bytes", None)

        if p and os.path.isfile(p):
            img = Image.open(p)
            img = img.convert("RGB") if force_rgb else img
            return img, p

        if b is not None:
            img = Image.open(io.BytesIO(b))
            img = img.convert("RGB") if force_rgb else img
            out_path = os.path.join(tmp_dir, f"{stem}.png")
            img.save(out_path)
            return img, out_path

    raise TypeError(f"Unsupported HF field type for image/mask: {type(field)}")


# ------------------------- evaluators -------------------------

def evaluate_json(args):
    if not os.path.isfile(args.input_json):
        raise FileNotFoundError(f"JSON file not found: {args.input_json}")

    with open(args.input_json, "r") as f:
        payload = json.load(f)

    items = payload.get("items", [])
    if args.limit is not None:
        items = items[: args.limit]

    base_dir = os.path.dirname(os.path.abspath(args.input_json))
    if args.save_preds:
        _ensure_dir(args.save_preds)

    predictor, plm = _build_predictor_and_plm(
        args.model_cfg, args.base_ckpt, args.final_ckpt, args.plm_ckpt,
        args.lora_ckpt, args.device, precision=args.precision
    )
    model = SAM2PLMWrapper(predictor, plm, device=args.device, precision=args.precision)
    model.eval()

    total_inter, total_union = 0.0, 0.0
    ious = []
    total_transposed = 0

    logger.info("Evaluating %d item(s) from %s", len(items), payload.get("dataset", "JSON"))
    for it in tqdm(items, desc="JSON items"):
        iid = it.get("id", "")
        img_p = _resolve_path(base_dir, it["image"])
        msk_p = _resolve_path(base_dir, it["mask"])
        prompt = it["prompt"]

        # read original image & mask in ORIGINAL resolution
        rgb_orig = _pil_read_rgb(img_p)
        gt = _read_mask(msk_p)

        H_img, W_img = rgb_orig.shape[:2]
        H_gt, W_gt = gt.shape[:2]
        if (H_img, W_img) != (H_gt, W_gt):
            raise RuntimeError(
                f"Shape mismatch for {iid}: image {W_img}x{H_img}, mask {W_gt}x{H_gt}"
            )

        with torch.no_grad():
            logit = model.infer_one(rgb_orig, prompt, img_p)  # [H, W] in original size

        H_pred, W_pred = logit.shape[-2], logit.shape[-1]
        if (H_pred, W_pred) != (H_img, W_img):
            raise RuntimeError(
                f"Predicted logit shape {W_pred}x{H_pred} does not match image size {W_img}x{H_img}"
            )

        gt_bool = torch.from_numpy((gt > 0).astype(np.bool_))  # CPU bool
        pred_bool = (logit.detach().to("cpu") > 0)             # CPU bool

        if pred_bool.shape == (gt_bool.shape[1], gt_bool.shape[0]) and pred_bool.shape != gt_bool.shape:
            pred_bool = pred_bool.T
            total_transposed += 1

        if pred_bool.shape != gt_bool.shape:
            raise RuntimeError(f"Shape mismatch for {iid}: pred {pred_bool.shape}, gt {gt_bool.shape}")

        inter, union, iou = _compute_iou(pred_bool, gt_bool)
        total_inter += inter
        total_union += union
        ious.append(iou)

        tqdm.write(f"{iid}: IoU {iou:.3f}")

        if args.save_preds:
            base_name = iid or os.path.splitext(os.path.basename(img_p))[0]

            pred_path = os.path.join(args.save_preds, f"{base_name}_pred.png")
            cv2.imwrite(pred_path, (pred_bool.numpy().astype(np.uint8) * 255))

            gt_path = os.path.join(args.save_preds, f"{base_name}_gt.png")
            cv2.imwrite(gt_path, (gt_bool.numpy().astype(np.uint8) * 255))

            panel_path = os.path.join(args.save_preds, f"{base_name}_panel.png")
            try:
                panel_img = compose_overlay(
                    img_path=Path(img_p),
                    mask_path=Path(pred_path),
                    tgt_w=1, tgt_h=1,
                    alpha_fill=0.70,
                    edge_w=2,
                    draw_box=False,
                    dpi=300,
                    no_resize=True,
                )
                panel_img.save(panel_path, dpi=(300, 300))
            except Exception as e:
                logger.warning("Failed to create overlay panel for %s: %s", iid, e)

            orig_out_path = os.path.join(args.save_preds, f"{base_name}_orig.png")
            try:
                with Image.open(img_p) as pil_img:
                    pil_img = pil_img.convert("RGB")
                    w_img, h_img = pil_img.size
                    radius = max(12, int(0.06 * min(w_img, h_img)))
                    styled_orig = _apply_rounded_corners(pil_img, radius)
                    styled_orig.save(orig_out_path, dpi=(300, 300))
            except Exception as e:
                logger.warning("Failed to save styled original for %s: %s", iid, e)

            prompt_out_path = os.path.join(args.save_preds, f"{base_name}_prompt.txt")
            try:
                with open(prompt_out_path, "w", encoding="utf-8") as f_txt:
                    f_txt.write(prompt)
            except Exception as e:
                logger.warning("Failed to save prompt txt for %s: %s", iid, e)

    if len(ious) == 0:
        logger.error("No valid items evaluated.")
        return

    logger.info("Total transposed predictions: %d out of %d", total_transposed, len(items))

    giou = float(np.mean(ious))
    ciou = float(total_inter / (total_union + 1e-10))

    logger.info("RESULTS (JSON mode): gIoU %.3f | cIoU %.3f", giou, ciou)
    print(f"JSON: giou{giou:.3f}_ciou{ciou:.3f}")


def evaluate_hf(args):
    if load_dataset is None:
        raise RuntimeError(
            "Hugging Face `datasets` is not available. Install with:\n"
            "  pip install datasets\n"
            f"Import error: {_HF_IMPORT_ERROR}"
        )

    if args.save_preds:
        _ensure_dir(args.save_preds)

    tmp_dir = tempfile.mkdtemp(prefix="convseg_eval_")

    # Load HF dataset (downloads/caches automatically)
    dsdict = load_dataset(
        args.hf_dataset,
        name=args.hf_config,
        cache_dir=args.hf_cache_dir,
    )

    # Ensure image/mask are decodable as images even if the dataset stores them as strings
    for sp in list(dsdict.keys()):
        cols = set(dsdict[sp].column_names)
        if "image" in cols and HFImage is not None:
            try:
                dsdict[sp] = dsdict[sp].cast_column("image", HFImage())
            except Exception:
                pass
        if "mask" in cols and HFImage is not None:
            try:
                dsdict[sp] = dsdict[sp].cast_column("mask", HFImage())
            except Exception:
                pass

    splits = [s.strip() for s in args.hf_splits.split(",") if s.strip()]
    missing = [s for s in splits if s not in dsdict]
    if missing:
        raise ValueError(f"Requested split(s) not found: {missing}. Available: {list(dsdict.keys())}")

    # Build model once
    predictor, plm = _build_predictor_and_plm(
        args.model_cfg, args.base_ckpt, args.final_ckpt, args.plm_ckpt,
        args.lora_ckpt, args.device, precision=args.precision
    )
    model = SAM2PLMWrapper(predictor, plm, device=args.device, precision=args.precision)
    model.eval()

    for split_name in splits:
        ds = dsdict[split_name]
        n_total = len(ds)
        n_eval = min(n_total, args.limit) if args.limit is not None else n_total

        logger.info("HF dataset=%s | config=%s | split=%s | evaluating %d/%d",
                    args.hf_dataset, args.hf_config, split_name, n_eval, n_total)

        total_inter, total_union = 0.0, 0.0
        ious = []
        total_transposed = 0

        for idx in tqdm(range(n_eval), desc=f"HF:{split_name}"):
            ex = ds[idx]
            iid = ex.get("id", f"{split_name}_{idx}")
            prompt = ex["prompt"]

            img_pil, img_path = _hf_field_to_pil_and_path(ex["image"], tmp_dir, stem=f"{iid}_img", force_rgb=True)
            msk_pil, _ = _hf_field_to_pil_and_path(ex["mask"], tmp_dir, stem=f"{iid}_gt", force_rgb=False)

            rgb_orig = np.array(img_pil.convert("RGB"), dtype=np.uint8)
            gt_u8 = _mask_pil_to_u8(msk_pil)

            H_img, W_img = rgb_orig.shape[:2]
            H_gt, W_gt = gt_u8.shape[:2]
            if (H_img, W_img) != (H_gt, W_gt):
                raise RuntimeError(
                    f"Shape mismatch for {iid}: image {W_img}x{H_img}, mask {W_gt}x{H_gt}"
                )

            with torch.no_grad():
                logit = model.infer_one(rgb_orig, prompt, img_path)

            H_pred, W_pred = logit.shape[-2], logit.shape[-1]
            if (H_pred, W_pred) != (H_img, W_img):
                raise RuntimeError(
                    f"Pred logit shape {W_pred}x{H_pred} does not match image size {W_img}x{H_img}"
                )

            gt_bool = torch.from_numpy((gt_u8 > 0).astype(np.bool_))
            pred_bool = (logit.detach().to("cpu") > 0)

            if pred_bool.shape == (gt_bool.shape[1], gt_bool.shape[0]) and pred_bool.shape != gt_bool.shape:
                pred_bool = pred_bool.T
                total_transposed += 1

            if pred_bool.shape != gt_bool.shape:
                raise RuntimeError(f"Shape mismatch for {iid}: pred {pred_bool.shape}, gt {gt_bool.shape}")

            inter, union, iou = _compute_iou(pred_bool, gt_bool)
            total_inter += inter
            total_union += union
            ious.append(iou)

            if args.save_preds:
                base_name = iid

                pred_path = os.path.join(args.save_preds, f"{base_name}_pred.png")
                cv2.imwrite(pred_path, (pred_bool.numpy().astype(np.uint8) * 255))

                gt_path = os.path.join(args.save_preds, f"{base_name}_gt.png")
                cv2.imwrite(gt_path, (gt_bool.numpy().astype(np.uint8) * 255))

                orig_out_path = os.path.join(args.save_preds, f"{base_name}_orig.png")
                img_pil.convert("RGB").save(orig_out_path, dpi=(300, 300))

                panel_path = os.path.join(args.save_preds, f"{base_name}_panel.png")
                try:
                    panel_img = compose_overlay(
                        img_path=Path(orig_out_path),
                        mask_path=Path(pred_path),
                        tgt_w=1, tgt_h=1,
                        alpha_fill=0.70,
                        edge_w=2,
                        draw_box=False,
                        dpi=300,
                        no_resize=True,
                    )
                    panel_img.save(panel_path, dpi=(300, 300))
                except Exception as e:
                    logger.warning("Failed overlay for %s: %s", iid, e)

                prompt_out_path = os.path.join(args.save_preds, f"{base_name}_prompt.txt")
                with open(prompt_out_path, "w", encoding="utf-8") as f:
                    f.write(prompt)

        if not ious:
            logger.error("No samples evaluated for split %s.", split_name)
            continue

        giou = float(np.mean(ious))
        ciou = float(total_inter / (total_union + 1e-10))
        logger.info("RESULTS (HF split=%s): gIoU %.3f | cIoU %.3f | transposed=%d",
                    split_name, giou, ciou, total_transposed)
        print(f"HF[{split_name}]: giou{giou:.3f}_ciou{ciou:.3f}")


# ------------------------- main -------------------------

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.input_json is not None:
        evaluate_json(args)
    else:
        evaluate_hf(args)


if __name__ == "__main__":
    main()
