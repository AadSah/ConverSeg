import contextlib
import os
import random

import cv2
import numpy as np
import torch

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


ASSUME_RGB_INPUT = True  # set False if your images are already BGR

def seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@contextlib.contextmanager
def _suppress_stderr():
    """Last-resort silencer for noisy C-level libs (e.g., libpng via OpenCV)."""
    saved = os.dup(2)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)

def _pil_read_rgb(path):
    with Image.open(path) as im:
        # drop ICC profile so it never surfaces again if you re-save
        im.info.pop("icc_profile", None)
        im = im.convert("RGB")
        return np.array(im)

def _pil_read_gray(path):
    with Image.open(path) as im:
        im.info.pop("icc_profile", None)
        im = im.convert("L")
        return np.array(im)

def read_img_mask(img_p, msk_p, max_dim=1024, backend="pil"):
    """
    Reads RGB image + grayscale mask, returns both as EXACTLY (max_dim x max_dim).
    - Preserves aspect ratio via resize on the long side, then center-padding.
    - Image: INTER_LINEAR for upscaling, INTER_AREA for downscaling.
    - Mask:  INTER_NEAREST always (no label bleeding).
    backend: "pil" avoids libpng iCCP warnings; "cv2" uses OpenCV.
    """
    if backend == "pil":
        img = _pil_read_rgb(img_p);   assert img is not None, img_p
        msk = _pil_read_gray(msk_p);  assert msk is not None, msk_p
        # img already RGB
    else:  # "cv2"
        with _suppress_stderr():
            img = cv2.imread(img_p, cv2.IMREAD_COLOR)
            msk = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        assert img is not None, img_p
        assert msk is not None, msk_p
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # keep RGB convention

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise RuntimeError(f"Empty image: {img_p}")

    # Scale so the LONG side becomes max_dim
    scale = float(max_dim) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Choose interpolation: AREA for downscale, LINEAR for upscale
    interp_img = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp_img)
    msk = cv2.resize(msk, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Center-padding to (max_dim, max_dim)
    pad_w = max_dim - new_w
    pad_h = max_dim - new_h
    left   = pad_w // 2
    right  = pad_w - left
    top    = pad_h // 2
    bottom = pad_h - top

    # Zero padding: image gets black, mask gets background 0
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    msk = cv2.copyMakeBorder(msk, top, bottom, left, right,
                             borderType=cv2.BORDER_CONSTANT, value=0)

    # Ensure contiguous arrays
    img = np.ascontiguousarray(img)
    msk = np.ascontiguousarray(msk)

    # Sanity check shapes
    assert img.shape[0] == max_dim and img.shape[1] == max_dim and img.ndim == 3, img.shape
    assert msk.shape[0] == max_dim and msk.shape[1] == max_dim and msk.ndim == 2, msk.shape
    return img, msk




def bin_mask(msk): return (msk > 0).astype(np.uint8)
def erode(m, k=5, it=1): return cv2.erode(m, np.ones((k,k), np.uint8), iterations=it) if k>0 and it>0 else m


def sample_points(m, n):
    ysx = np.argwhere(m > 0)
    if ysx.size == 0: return np.zeros((0,1,2), np.float32)
    pts = []
    for _ in range(max(n,1)):
        y,x = ysx[random.randrange(len(ysx))]
        pts.append([[float(x), float(y)]])
    return np.asarray(pts, np.float32)

def iou(pred, tgt):
    pred = pred.bool(); tgt = tgt.bool()
    inter = (pred & tgt).sum((1,2)).float()
    union = (pred | tgt).sum((1,2)).float().clamp(min=1)
    return (inter/union).mean()

def split(samples, test_size=0.2, seed_=42):
    rng = np.random.default_rng(seed_); idx = rng.permutation(len(samples))
    n_test = max(1, int(len(samples)*test_size))
    test = [samples[i] for i in idx[:n_test]]
    train = [samples[i] for i in idx[n_test:]]
    return train, test

# ------- pretty val visuals -------

UI_BG = (18,18,18)       # dark canvas
UI_GUTTER = 16
UI_PAD = 16
UI_FONT = cv2.FONT_HERSHEY_SIMPLEX

# colors (BGR)
COL_TP = (120, 200, 120)   # green-ish
COL_FP = (80, 110, 255)    # orange/red-ish
COL_FN = (0, 215, 255)     # yellow
COL_EDGE = (255, 255, 255) # white
COL_TEXT = (240,240,240)
COL_SHADOW = (0,0,0)

def to_uint8_3ch(img):
    im = img
    if im.dtype != np.uint8:
        im = im.astype(np.float32)
        if im.max() <= 1.0: im *= 255.0
        im = np.clip(im, 0, 255).astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im

def draw_text(img, text, org, scale=0.7, color=COL_TEXT, thick=2):
    cv2.putText(img, text, org, UI_FONT, scale, COL_SHADOW, thick+2, cv2.LINE_AA)
    cv2.putText(img, text, org, UI_FONT, scale, color, thick, cv2.LINE_AA)

def banner(img, left_text, right_text=None, alpha=0.65):
    """semi-transparent banner across top with left/right text"""
    h = 40  # banner height
    overlay = img.copy().astype(np.float32)
    base = img.astype(np.float32)
    overlay[:h] = (40,40,40)  # dark bar
    out = (alpha*overlay + (1-alpha)*base).astype(np.uint8)

    # left text
    draw_text(out, left_text, (12, 26), scale=0.7)

    # right text (align from right edge)
    if right_text:
        ((tw, th), _) = cv2.getTextSize(right_text, UI_FONT, 0.7, 2)
        draw_text(out, right_text, (out.shape[1]-tw-12, 26), scale=0.7)
    return out

def confusion_overlay(base_bgr, gt_bool, pred_bool, alpha=0.40, draw_edges=True):
    """TP=green, FP=red/orange, FN=yellow; optional white outline for pred."""
    img = to_uint8_3ch(base_bgr)
    gt = gt_bool.astype(bool)
    pr = pred_bool.astype(bool)

    tp = gt & pr
    fp = (~gt) & pr
    fn = gt & (~pr)

    out = img.astype(np.float32)
    for m, col in [(tp, COL_TP), (fp, COL_FP), (fn, COL_FN)]:
        if m.any():
            out[m] = alpha*np.array(col, dtype=np.float32) + (1.0-alpha)*out[m]

    if draw_edges and pr.any():
        # crisp edge via contours
        m8 = (pr.astype(np.uint8))*255
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COL_EDGE, 2, lineType=cv2.LINE_AA)

    return np.clip(out, 0, 255).astype(np.uint8)

def legend(img, items=(("TP", COL_TP), ("FP", COL_FP), ("FN", COL_FN))):
    """small legend pills in top-right below banner"""
    x = img.shape[1] - 12
    y = 46
    for label, col in items:
        # pill
        box_w = 18
        x -= (box_w + 8 + 6 + cv2.getTextSize(label, UI_FONT, 0.6, 2)[0][0])
        cv2.rectangle(img, (x, y-10), (x+box_w, y+2), col, thickness=-1)
        draw_text(img, label, (x+box_w+6, y), scale=0.6)
        x -= 16
    return img

def pad_bg(im, pad=UI_PAD, bg=UI_BG):
    return cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)

def hcat_gutter(a, b, gutter=UI_GUTTER, bg=UI_BG):
    H = max(a.shape[0], b.shape[0])
    W = a.shape[1] + gutter + b.shape[1]
    out = np.full((H, W, 3), bg, dtype=np.uint8)
    out[:a.shape[0], :a.shape[1]] = a
    out[:b.shape[0], a.shape[1]+gutter:a.shape[1]+gutter+b.shape[1]] = b
    return out

def vstack_rows(rows, gutter=UI_GUTTER, bg=UI_BG):
    W = max(r.shape[1] for r in rows)
    H = sum(r.shape[0] for r in rows) + gutter*(len(rows)-1)
    out = np.full((H, W, 3), bg, dtype=np.uint8)
    y = 0
    for r in rows:
        out[y:y+r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + gutter
    return out

def save_grid(rows, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vstack_rows(rows))


def put_text(img, s, y=28, x=10, max_width_ratio=0.7, alpha=0.65):
    f = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]
    scale = max(0.5, min(1.6, w/512*0.7))
    th = max(1, int(2*scale))
    max_w = int(w * max_width_ratio)

    # wrap (word-based)
    def tw(t): return cv2.getTextSize(t, f, scale, th)[0][0]
    lines, cur = [], ""
    for word in s.split():
        t = word if not cur else cur + " " + word
        if tw(t) <= max_w: cur = t
        else: lines.append(cur) if cur else None; cur = word
    if cur: lines.append(cur)

    sizes = [cv2.getTextSize(t, f, scale, th)[0] for t in lines]
    lh = max(s[1] for s in sizes)
    bw = max(s[0] for s in sizes)
    x1, y1 = max(0, x-8), max(0, y-lh-10)
    x2, y2 = min(w-1, x+bw+8), min(h-1, y+lh*(len(lines)-1)+8)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,0), cv2.FILLED)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    for i, t in enumerate(lines):
        yy = y + i*lh
        cv2.putText(img, t, (x, yy), f, scale, (0,0,0), th+2, cv2.LINE_AA)   # outline
        cv2.putText(img, t, (x, yy), f, scale, (255,255,255), th, cv2.LINE_AA)

def to_bgr_u8(img):
    x = img
    # to uint8
    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating) and x.max() <= 1.0:
            x = (np.clip(x, 0, 1) * 255.0).round().astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).round().astype(np.uint8)
    # to 3ch
    if x.ndim == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    elif ASSUME_RGB_INPUT:
        # standardize: RGB -> BGR for cv2 ops & imwrite
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x

def overlay_mask(bgr, mask_bool, color=(0,255,0), alpha=0.65, edge_px=1):
    # out = to_bgr_u8(bgr).astype(np.float32)
    out = bgr.astype(np.float32)
    m = mask_bool.astype(bool)
    if m.ndim == 3: m = m.squeeze(-1)
    if m.any():
        c = np.array(color, dtype=np.float32)
        out[m] = alpha*c + (1.0 - alpha)*out[m]
        if edge_px > 0:
            m8 = (m.astype(np.uint8))*255
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color, edge_px, lineType=cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8)
