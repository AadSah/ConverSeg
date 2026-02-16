#!/usr/bin/env python3
# minimal_sam2_plm_interactive_overlay_square.py

import os
import argparse
import logging
import hashlib
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import (
    Image,
    ImageFilter,
    ImageChops,
    ImageDraw,
)

# ----------------- overlay style helpers (from overlay_one.py, simplified) -----------------

EDGE_COLORS_HEX = ["#3A86FF", "#FF006E", "#43AA8B", "#F3722C", "#8338EC", "#90BE6D"]


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


EDGE_COLORS = [_hex_to_rgb(h) for h in EDGE_COLORS_HEX]


def stable_color(key: str):
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    return EDGE_COLORS[h % len(EDGE_COLORS)]


def tint(rgb, amt: float = 0.1):
    # soft pastel fill by mixing with white
    return tuple(int(255 - (255 - c) * (1 - amt)) for c in rgb)


def edge_map(mask_bool: np.ndarray, width_px: int = 2) -> Image.Image:
    m = Image.fromarray((mask_bool.astype(np.uint8) * 255), "L")
    edges = ImageChops.difference(
        m.filter(ImageFilter.MaxFilter(3)), m.filter(ImageFilter.MinFilter(3))
    )
    for _ in range(max(0, width_px - 1)):
        edges = edges.filter(ImageFilter.MaxFilter(3))
    return edges.point(lambda p: 255 if p > 0 else 0)


def _apply_rounded_corners(img_rgb: Image.Image, radius: int) -> Image.Image:
    w, h = img_rgb.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    bg = Image.new("RGB", (w, h), "white")
    img_rgba = img_rgb.convert("RGBA")
    img_rgba.putalpha(mask)
    bg.paste(img_rgba.convert("RGB"), (0, 0), mask)
    return bg


def make_overlay(rgb: np.ndarray, mask: np.ndarray, key: str = "mask") -> Image.Image:
    """
    rgb: HxWx3 uint8
    mask: HxW uint8 (0 or 255)
    """
    base = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    H, W = mask.shape[:2]
    if base.size != (W, H):
        base = base.resize((W, H), Image.BICUBIC)

    base_rgba = base.convert("RGBA")
    mask_bool = mask > 0

    color = stable_color(key)
    fill_rgb = tint(color, 0.1)
    alpha_fill = 0.7
    edge_width = 2
    draw_box = False

    a = int(round(alpha_fill * 255))
    tgt_w, tgt_h = base_rgba.size

    # fill layer
    fill_layer = Image.new("RGBA", (tgt_w, tgt_h), fill_rgb + (0,))
    fill_alpha = Image.fromarray((mask_bool.astype(np.uint8) * a), "L")
    fill_layer.putalpha(fill_alpha)

    # edge layer
    edgesL = edge_map(mask_bool, width_px=edge_width)
    stroke = Image.new("RGBA", (tgt_w, tgt_h), color + (0,))
    stroke.putalpha(edgesL)

    # composite
    out = Image.alpha_composite(base_rgba, fill_layer)
    out = Image.alpha_composite(out, stroke)

    # optional bounding box
    if draw_box:
        ys, xs = np.where(mask_bool)
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
            d.rectangle(
                [x0, y0, x1, y1],
                outline=tint(color, 0.80) + (140,),
                width=2,
            )
            out = Image.alpha_composite(out, box)

    out = out.convert("RGB")
    out = _apply_rounded_corners(out, max(12, int(0.06 * min(out.size))))
    return out


# ----------------- resize + pad helpers (match your eval wrapper) -----------------

SQUARE_DIM = 1024  # same as SAM2PLMWrapper.square_dim


def _pil_read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im.info.pop("icc_profile", None)
        im = im.convert("RGB")
        return np.array(im)


def _resize_pad_square(arr: np.ndarray, max_dim: int, *, is_mask: bool) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        raise RuntimeError("Empty array in _resize_pad_square")
    scale = float(max_dim) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if is_mask:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    arr = cv2.resize(arr, (new_w, new_h), interpolation=interp)

    pad_w = max_dim - new_w
    pad_h = max_dim - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    border_val = 0 if is_mask else (0, 0, 0)
    arr = cv2.copyMakeBorder(
        arr, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=border_val
    )
    return np.ascontiguousarray(arr)


def _resize_pad_square_meta(h: int, w: int, max_dim: int):
    scale = float(max_dim) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pad_w = max_dim - new_w
    pad_h = max_dim - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return {
        "scale": scale,
        "new_w": new_w,
        "new_h": new_h,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }


def _unpad_and_resize_pred_to_gt(
    logit_sq: torch.Tensor, meta: dict, out_hw: tuple[int, int]
) -> torch.Tensor:
    top, left = meta["top"], meta["left"]
    nh, nw = meta["new_h"], meta["new_w"]
    crop = logit_sq[top : top + nh, left : left + nw]  # [nh, nw]
    crop = crop.unsqueeze(0).unsqueeze(0)  # [1,1,nh,nw]
    up = F.interpolate(crop, size=out_hw, mode="bilinear", align_corners=False)
    return up[0, 0]  # [Hgt, Wgt]


# -------------------------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser("Interactive SAM2+PLM inference with overlay (square 1024)")
    p.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml")
    p.add_argument("--base_ckpt", type=str, default="./checkpoints/sam2_hiera_large.pt")
    p.add_argument("--final_ckpt", type=str, required=True, help="Fine-tuned SAM2 checkpoint (.torch)")
    p.add_argument("--plm_ckpt", type=str, required=True, help="PLM adapter checkpoint (.torch)")
    p.add_argument("--lora_ckpt", type=str, required=True, help="LoRA adapter path")
    p.add_argument("--image", type=str, default=None, help="Optional single image path for one-shot inference")
    p.add_argument("--prompt", type=str, default=None, help="Optional single text prompt for one-shot inference")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to save mask and overlay images")
    return p.parse_args()


def _dtype_from_precision(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def build_model(args):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise ImportError(
            "Could not import SAM2 modules. Install SAM2 in this environment with: `pip install -e ./sam2`."
        ) from e

    try:
        from models.language_adapter import LanguageAdapter
    except ImportError as e:
        raise ImportError(
            "Could not import `models.language_adapter`. Run demo.py from the ConverSeg repo root."
        ) from e

    device = args.device

    if not os.path.isfile(args.base_ckpt):
        raise FileNotFoundError(f"Could not find SAM2 base checkpoint: {args.base_ckpt}")
    if not os.path.isfile(args.final_ckpt):
        raise FileNotFoundError(f"Could not find fine-tuned SAM2 checkpoint: {args.final_ckpt}")
    if not os.path.isfile(args.plm_ckpt):
        raise FileNotFoundError(f"Could not find PLM adapter checkpoint: {args.plm_ckpt}")
    if args.lora_ckpt is not None and not os.path.exists(args.lora_ckpt):
        raise FileNotFoundError(f"Could not find LoRA checkpoint: {args.lora_ckpt}")

    plm_dtype = _dtype_from_precision(args.precision)
    if device == "cpu" and plm_dtype != torch.float32:
        logging.warning("FP16/BF16 on CPU is not supported reliably. Falling back to fp32 for PLM.")
        plm_dtype = torch.float32

    model = build_sam2(args.model_cfg, args.base_ckpt, device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.model.eval()

    sd = torch.load(args.final_ckpt, map_location=device)
    predictor.model.load_state_dict(sd.get("model", sd), strict=True)

    C = predictor.model.sam_mask_decoder.transformer_dim
    plm = LanguageAdapter(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        transformer_dim=C,
        n_sparse_tokens=0,
        use_dense_bias=True,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        dtype=plm_dtype,
        device=device,
        use_image_input=True,
    ).to(device)

    plm_sd = torch.load(args.plm_ckpt, map_location=device)
    plm_state = plm_sd["plm"] if isinstance(plm_sd, dict) and "plm" in plm_sd else plm_sd
    plm.load_state_dict(plm_state, strict=True)
    if args.lora_ckpt is not None:
        plm.load_lora(args.lora_ckpt)
    plm.eval()

    return predictor, plm


def get_text_to_image_attention(decoder: Any):
    """
    Returns a tensor of token->image cross-attention maps from the SAM decoder's transformer.

    Shape:
      layers+final, B, heads, N_text_tokens, N_image_tokens
    """
    two_way = getattr(decoder, "transformer", None)
    if two_way is None:
        return None

    # Collect cross-attn (tokens -> image) from each block
    attn_blocks = []
    for blk in getattr(two_way, "layers", []):
        attn_mod = getattr(blk, "cross_attn_token_to_image", None)
        a = getattr(attn_mod, "last_attn", None)  # [B, H, N_q, N_k]
        if a is not None:
            attn_blocks.append(a)

    # And the final token->image attention (same Attention class)
    final_mod = getattr(two_way, "final_attn_token_to_image", None)
    final = getattr(final_mod, "last_attn", None)
    if final is not None:
        attn_blocks.append(final)

    if not attn_blocks:
        return None  # nothing was recorded

    # Stack over layers: [L, B, heads, N_q, N_k]
    attn = torch.stack(attn_blocks, dim=0)

    # Figure out how many tokens at the beginning are *output tokens*
    # MaskDecoder.predict_masks builds:
    #   output_tokens = [iou_token, mask_tokens]  (plus obj_score_token if pred_obj_scores)
    #   tokens = cat(output_tokens, sparse_prompt_embeddings, dim=1)
    s = 1 if getattr(decoder, "pred_obj_scores", False) else 0
    n_output_tokens = s + 1 + int(getattr(decoder, "num_mask_tokens", 0))

    # Everything after that in the query dimension is from sparse_prompt_embeddings
    # (in your case: text tokens)
    if attn.shape[-2] <= n_output_tokens:
        return None
    text_attn = attn[..., n_output_tokens:, :]  # [L, B, heads, N_text, N_image_tokens]

    return text_attn


@torch.no_grad()
def run_inference(predictor, plm, image_path, text):
    # Read ORIGINAL image
    rgb_orig = _pil_read_rgb(image_path)
    Hgt, Wgt = rgb_orig.shape[:2]
    meta = _resize_pad_square_meta(Hgt, Wgt, SQUARE_DIM)

    # Square-pad only for the model
    rgb_sq = _resize_pad_square(rgb_orig, SQUARE_DIM, is_mask=False)

    predictor.set_image(rgb_sq)
    image_emb = predictor._features["image_embed"][-1].unsqueeze(0)  # [1,C,H_feat,W_feat]
    hi = [lvl[-1].unsqueeze(0) for lvl in predictor._features["high_res_feats"]]
    _, _, H_feat, W_feat = image_emb.shape

    # PLM prompts
    sp, dp = plm([text], H_feat, W_feat, [image_path])

    dec = predictor.model.sam_mask_decoder
    dev, dtype = next(dec.parameters()).device, next(dec.parameters()).dtype
    image_pe = predictor.model.sam_prompt_encoder.get_dense_pe().to(dev, dtype)
    image_emb = image_emb.to(dev, dtype)
    hi = [h.to(dev, dtype) for h in hi]
    sp, dp = sp.to(dev, dtype), dp.to(dev, dtype)

    # --- SAM2 decoding ---
    low, scores, _, _ = dec(
        image_embeddings=image_emb,
        image_pe=image_pe,
        sparse_prompt_embeddings=sp,
        dense_prompt_embeddings=dp,
        multimask_output=True,
        repeat_image=False,
        high_res_features=hi,
    )

    # First go to square space (1024x1024), then unpad+resize to original image size
    logits_sq = predictor._transforms.postprocess_masks(
        low, (SQUARE_DIM, SQUARE_DIM)
    )  # [1,3,sq,sq]
    best = scores.argmax(dim=1).item()
    logit_sq = logits_sq[0, best]  # [sq, sq]
    logit_gt = _unpad_and_resize_pred_to_gt(logit_sq, meta, (Hgt, Wgt))  # [Hgt, Wgt]

    prob = torch.sigmoid(logit_gt)
    mask = (prob > 0.5).cpu().numpy().astype(np.uint8) * 255

    # ---- GLOBAL text->image attention map ----
    text_attn = get_text_to_image_attention(dec)  # [L, B, heads, N_text, N_img] or None

    global_attn_orig = None
    if text_attn is not None:
        # text_attn: [L, B, H_heads, N_text, N_img]
        L, B, H_heads, N_text, N_img = text_attn.shape
        if B != 1 or N_text == 0 or N_img != (H_feat * W_feat):
            return rgb_orig, mask, None

        # Average over layers, heads, and text tokens -> [B, N_img]
        # dims: 0=L, 2=H_heads, 3=N_text
        attn_flat = text_attn.mean(dim=(0, 2, 3))  # [B, N_img]
        global_flat = attn_flat[0]                 # [N_img]

        # Reshape to feature map
        a = global_flat.view(H_feat, W_feat)       # [H_feat, W_feat]

        # Normalize to [0,1]
        a = a - a.min()
        denom = a.max().clamp(min=1e-6)
        a = a / denom

        # Upsample to (SQUARE_DIM, SQUARE_DIM)
        a_sq = F.interpolate(
            a.unsqueeze(0).unsqueeze(0),
            size=(SQUARE_DIM, SQUARE_DIM),
            mode="bilinear",
            align_corners=False,
        )[0, 0]  # [SQUARE_DIM, SQUARE_DIM]

        # Unpad & resize back to original image size
        a_gt = _unpad_and_resize_pred_to_gt(a_sq, meta, (Hgt, Wgt))  # [Hgt, Wgt]
        global_attn_orig = a_gt.cpu().numpy()  # float [Hgt, Wgt] in ~[0,1]

    return rgb_orig, mask, global_attn_orig


def make_pretty_mask(mask: np.ndarray, key: str) -> Image.Image:
    """
    Take a binary 0/255 mask [H,W] and return a nice colored mask on white background.
    """
    assert mask.ndim == 2
    h, w = mask.shape
    color = stable_color(key)  # (r,g,b)

    bg = np.ones((h, w, 3), dtype=np.uint8) * 255  # white
    fg = np.zeros_like(bg, dtype=np.uint8)
    fg[..., 0] = color[0]
    fg[..., 1] = color[1]
    fg[..., 2] = color[2]

    m = (mask > 0)[..., None]
    out = np.where(m, fg, bg)
    return Image.fromarray(out, mode="RGB")


def make_attn_overlay(rgb: np.ndarray, attn: np.ndarray, alpha: float = 0.6) -> Image.Image:
    """
    Overlay a colored attention heatmap onto the RGB image.

    rgb: HxWx3 uint8
    attn: HxW float (any range, will be normalized)
    """
    h, w = rgb.shape[:2]
    ah, aw = attn.shape
    if (ah, aw) != (h, w):
        attn_resized = cv2.resize(attn.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        attn_resized = attn.astype(np.float32)

    # Normalize attn -> [0, 255]
    attn_resized = attn_resized - attn_resized.min()
    denom = attn_resized.max()
    if denom < 1e-6:
        denom = 1e-6
    attn_norm = (attn_resized / denom * 255.0).clip(0, 255).astype(np.uint8)

    # Colorize with a heatmap
    heatmap_bgr = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original RGB
    rgb_f = rgb.astype(np.float32)
    heat_f = heatmap_rgb.astype(np.float32)
    blended = (1.0 - alpha) * rgb_f + alpha * heat_f
    blended = blended.clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended, mode="RGB")


def save_outputs(rgb, mask, global_attn, image_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    # Base paths
    out_mask = os.path.join(out_dir, f"{stem}_mask.png")
    out_mask_color = os.path.join(out_dir, f"{stem}_mask_color.png")
    out_overlay = os.path.join(out_dir, f"{stem}_overlay.png")
    out_attn_gray = os.path.join(out_dir, f"{stem}_attn_global_gray.png")
    out_attn_overlay = os.path.join(out_dir, f"{stem}_attn_global_overlay.png")

    # 1) Raw binary mask (unchanged)
    Image.fromarray(mask, mode="L").save(out_mask)

    # 2) Pretty colored mask
    pretty_mask_img = make_pretty_mask(mask, key=image_path)
    pretty_mask_img.save(out_mask_color, dpi=(300, 300))

    # 3) Segmentation overlay (you already had this, still nice)
    overlay_img = make_overlay(rgb, mask, key=image_path)
    overlay_img.save(out_overlay, dpi=(300, 300))

    print(f"Saved binary mask to:       {out_mask}")
    print(f"Saved colored mask to:      {out_mask_color}")
    print(f"Saved segmentation overlay: {out_overlay}")

    # 4) Attention visualizations (if available)
    if global_attn is not None:
        # grayscale heatmap
        a = global_attn.astype(np.float32)
        a = a - a.min()
        denom = a.max()
        if denom < 1e-6:
            denom = 1e-6
        a_norm = (a / denom * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(a_norm, mode="L").save(out_attn_gray, dpi=(300, 300))

        # colored overlay on RGB
        attn_overlay_img = make_attn_overlay(rgb, global_attn)
        attn_overlay_img.save(out_attn_overlay, dpi=(300, 300))

        print(f"Saved global attention map (gray) to:      {out_attn_gray}")
        print(f"Saved global attention overlay (pretty) to: {out_attn_overlay}")
    else:
        print("No attention maps recorded (global_attn is None).")




def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    one_shot_mode = (args.image is not None) or (args.prompt is not None)
    if one_shot_mode and (not args.image or not args.prompt):
        raise ValueError("When using one-shot mode, both --image and --prompt are required.")

    logging.info("Loading model once...")
    predictor, plm = build_model(args)
    logging.info("Model ready.")

    if one_shot_mode:
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        prompt = args.prompt.strip() or "segment"
        rgb, mask, global_attn = run_inference(predictor, plm, args.image, prompt)
        save_outputs(rgb, mask, global_attn, args.image, args.out_dir)
        print("Done.")
        return

    logging.info("Entering interactive mode. Type 'quit' to exit.")

    while True:
        img_path = input("\nImage path (or 'quit' to exit): ").strip()
        if img_path.lower() in {"quit", "q", "exit"}:
            break
        if not os.path.isfile(img_path):
            print("Image not found, try again.")
            continue

        prompt = input("Text prompt (or 'quit' to exit): ").strip()
        if prompt.lower() in {"quit", "q", "exit"}:
            break
        prompt = prompt or "segment"

        try:
            rgb, mask, global_attn = run_inference(predictor, plm, img_path, prompt)
            save_outputs(rgb, mask, global_attn, img_path, args.out_dir)
        except Exception as e:
            print(f"Error: {e}")


    print("Done.")


if __name__ == "__main__":
    main()
