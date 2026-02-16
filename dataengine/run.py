import os, re, json, gc, time
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from contextlib import contextmanager
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM, Sam2Processor, Sam2Model
import hydra
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from google import genai
from google.genai import types
import shutil
from time import sleep


_GEMINI_CLIENT = genai.Client()
_GEMINI_MODEL = "gemini-flash-latest"

try:
    from meta_prompts.region_dense_caption_meta_prompt import REGION_DENSE_CAPTION_META_PROMPT
except Exception:
    REGION_DENSE_CAPTION_META_PROMPT = (
        "You are an expert vision-language model. Given an image, produce a numbered list of concise, non-overlapping, "
        "visually grounded region descriptions. Each line must be in the format: [<index>: <short region caption>]. "
        "Order regions from most salient to least. Be brief."
    )

from meta_prompts.concept_specific_meta_prompts import ENTITIES_META_PROMPT, SPATIAL_META_PROMPT, AFFORDANCES_META_PROMPT, RELATIONS_META_PROMPT, PHYSICS_META_PROMPT

CONCEPT_META_PROMPTS = {
    "ENTITIES": ENTITIES_META_PROMPT,
    "SPATIAL": SPATIAL_META_PROMPT,
    "AFFORDANCES": AFFORDANCES_META_PROMPT,
    "RELATIONS": RELATIONS_META_PROMPT,
    "PHYSICS": PHYSICS_META_PROMPT,
}

# --- cost tracking (env-configurable USD estimation) ---
_G_PRICE_IN  = float(os.getenv("GEMINI_PRICE_IN_PER_1M", "0.30"))
_G_PRICE_OUT = float(os.getenv("GEMINI_PRICE_OUT_PER_1M", "2.50"))

_COST = {"in": 0, "out": 0, "usd": 0.0, "calls": 0}


# cache for SAM2AutomaticMaskGenerator keyed by (cfg, ckpt, device, pps, nms)
_SAM2_AUTOGEN = {}


class StepTimer:
    def __init__(self):
        self._totals = defaultdict(float)
        self._counts = defaultdict(int)
        self._records: List[Dict[str, float]] = []

    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._totals[name] += duration
            self._counts[name] += 1
            self._records.append({"step": name, "duration": duration})
            print(f"[TIMER] {name} took {duration:.3f}s")

    def summary(self) -> List[Dict[str, float]]:
        ordered_steps = sorted(self._totals.keys(), key=lambda k: self._totals[k], reverse=True)
        return [
            {
                "step": step,
                "total_seconds": self._totals[step],
                "count": self._counts[step],
                "avg_seconds": self._totals[step] / self._counts[step] if self._counts[step] else 0.0,
            }
            for step in ordered_steps
        ]

    def report(self) -> List[Dict[str, float]]:
        ordered = self.summary()
        if not ordered:
            return ordered
        print("[TIMER] ---- step totals ----")
        for item in ordered:
            print(
                f"[TIMER] {item['step']}: total={item['total_seconds']:.3f}s "
                f"count={item['count']} avg={item['avg_seconds']:.3f}s"
            )
        slowest = ordered[0]
        print(f"[TIMER] Slowest step overall: {slowest['step']} ({slowest['total_seconds']:.3f}s total)")
        return ordered


def _cost_reset():
    _COST.update({"in": 0, "out": 0, "usd": 0.0, "calls": 0})

def _cost_log(resp, tag=""):
    um = getattr(resp, "usage_metadata", None)
    if not um: return
    pin  = getattr(um, "prompt_token_count", 0) or getattr(um, "input_tokens", 0) or 0
    pout = getattr(um, "candidates_token_count", 0) or getattr(um, "output_tokens", 0) or 0
    usd  = (pin * _G_PRICE_IN + pout * _G_PRICE_OUT) / 1_000_000.0 if (_G_PRICE_IN or _G_PRICE_OUT) else 0.0
    _COST["in"] += pin; _COST["out"] += pout; _COST["usd"] += usd; _COST["calls"] += 1
    print(f"[GEMINI] {tag} in={pin} out={pout}" + (f" cost=${usd:.6f}" if usd else ""))


def _device(): return "cuda" if torch.cuda.is_available() else "cpu"
def _sanitize(s:str)->str: return re.sub(r"[^a-zA-Z0-9._\-]+","_",s)[:80]
def _ensure_hw(m,H,W):
    a=m.detach().cpu().numpy() if hasattr(m,"detach") else np.asarray(m)
    a=np.squeeze(a)
    if a.ndim==1 and a.size==H*W: a=a.reshape(H,W)
    if a.ndim==3 and 1 in a.shape: a=np.squeeze(a)
    if a.shape==(W,H): a=a.T
    if a.shape!=(H,W): raise ValueError(f"{a.shape}!={(H,W)}")
    return a.astype(bool)

# def _mask_center(mask): 
#     if mask is None: return None
#     ys,xs=np.nonzero(mask)
#     if len(xs)==0: return None
#     return int(xs.mean()),int(ys.mean())

def _to_gemini_image(img: Image.Image, max_dim=768) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / float(max(w, h))
    new = (int(round(w*scale)), int(round(h*scale)))
    return img.resize(new, Image.LANCZOS)


def _mask_center(mask, connectivity=4, method="median"):
    if mask is None: return None
    m = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    m = np.squeeze(m).astype(bool)
    if m.ndim != 2 or not m.any(): return None
    H, W = m.shape
    vis = np.zeros_like(m, bool)
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)] if connectivity == 4 else [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    best = ([], [])
    for y0, x0 in zip(*np.nonzero(m)):
        if vis[y0, x0]: continue
        stack, xs, ys = [(y0, x0)], [], []
        vis[y0, x0] = True
        while stack:
            y, x = stack.pop(); xs.append(x); ys.append(y)
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and not vis[ny, nx]:
                    vis[ny, nx] = True; stack.append((ny, nx))
        if len(xs) > len(best[0]): best = (xs, ys)
    if not best[0]: return None
    xs, ys = np.array(best[0]), np.array(best[1])
    if method == "mean": cx, cy = int(round(xs.mean())), int(round(ys.mean()))
    else:               cx, cy = int(np.median(xs)), int(np.median(ys))
    return (cx, cy)


def _box_center(box):
    if not box: return None
    x0,y0,x1,y1=box
    return (x0+x1)//2,(y0+y1)//2

def _draw_mark(im,pt,idx):
    if not pt: return im
    x,y=pt; im=im.copy(); d=ImageDraw.Draw(im,"RGBA")
    t=str(idx); b=d.textbbox((0,0),t); tw,th=b[2]-b[0],b[3]-b[1]
    pad,r=4,6
    box=[x-(tw+2*pad)//2,y-(th+2*pad)//2,x+(tw+2*pad)//2,y+(th+2*pad)//2]
    d.rectangle(box,fill=(0,0,0,230),outline=(255,255,255,160),width=1)
    d.text((x-tw//2,y-th//2),t,fill=(255,255,255,255))
    return im

def _mask_bbox(mask):
    ys,xs=np.nonzero(mask.astype(bool))
    if xs.size==0: return None
    return [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]

def _draw_bbox(im,box):
    if not box: return im
    im=im.copy(); d=ImageDraw.Draw(im,"RGBA")
    d.rectangle(box,outline=(0,255,0,255),width=2)
    d.rectangle([box[0]-1,box[1]-1,box[2]+1,box[3]+1],outline=(0,0,0,180),width=1)
    return im

# def _overlay_mask(img,mask,alpha=0.45):
#     base=img.convert("RGBA"); W,H=base.size
#     arr=np.zeros((H,W,4),dtype=np.float32); sl=_ensure_hw(mask,H,W)
#     arr[sl]=np.array([255,0,0,alpha*255],dtype=np.float32)
#     ov=Image.fromarray(np.clip(arr,0,255).astype(np.uint8),"RGBA")
#     return Image.alpha_composite(base,ov).convert("RGB")

def _overlay_mask(img,mask,alpha=0.45,dim=0.6):
    base=img.convert("RGBA"); W,H=base.size
    arr=np.zeros((H,W,4),dtype=np.float32); sl=_ensure_hw(mask,H,W)
    arr[...,3]=dim*255  # dim unmasked region
    arr[sl]=np.array([255,0,0,alpha*255],dtype=np.float32)
    ov=Image.fromarray(np.clip(arr,0,255).astype(np.uint8),"RGBA")
    return Image.alpha_composite(base,ov).convert("RGB")

def _overlay_mask_transparent(img,mask,alpha=0.45,dim=0.6):
    base=img.convert("RGBA"); W,H=base.size
    arr=np.zeros((H,W,4),dtype=np.float32); sl=_ensure_hw(mask,H,W)
    arr[...,3]=dim*255          # dim everywhere
    arr[sl]=0                   # but NOT over the mask
    ov=Image.fromarray(np.clip(arr,0,255).astype(np.uint8),"RGBA")
    return Image.alpha_composite(base,ov).convert("RGB")

def _overlay_with_bbox(img,mask,mark_pt_idx=None):
    im=_overlay_mask(img,mask)
    b=_mask_bbox(mask)
    im=_draw_bbox(im,b)
    # if mark_pt_idx:
    #     pt,idx=(mark_pt_idx[0],mark_pt_idx[1]),mark_pt_idx[2]
    #     im=_draw_mark(im,pt,idx)
    return im,b

def _overlay_with_bbox_without_mask(img,mask,mark_pt_idx=None):
    im=_overlay_mask_transparent(img,mask)
    b=_mask_bbox(mask)
    im=_draw_bbox(im,b)
    # if mark_pt_idx:
    #     pt,idx=(mark_pt_idx[0],mark_pt_idx[1]),mark_pt_idx[2]
    #     im=_draw_mark(im,pt,idx)
    return im,b

def build_auto_mask_generator(cfg, ckpt, device=None, pps=32, nms=0.7):
    key = (cfg, ckpt, device or _device(), pps, nms)
    if key in _SAM2_AUTOGEN:
        return _SAM2_AUTOGEN[key]
    gh = hydra.core.global_hydra.GlobalHydra.instance()
    if not gh.is_initialized():
        hydra.initialize_config_module('sam2/configs/sam2_1', version_base='1.2')
    sam2 = build_sam2(cfg, ckpt, device=device or _device(), apply_postprocessing=False)
    gen = SAM2AutomaticMaskGenerator(sam2, points_per_side=pps, box_nms_thresh=nms, output_mode="binary_mask")
    _SAM2_AUTOGEN[key] = gen
    return gen

_GEMMA_MODEL=_GEMMA_PROC=_MD_MODEL=_SAM2_T_MODEL=_SAM2_T_PROC=None
def _load_gemma(mid="google/gemma-3-27b-it"):
    global _GEMMA_MODEL,_GEMMA_PROC
    if _GEMMA_MODEL is None:
        _GEMMA_PROC=AutoProcessor.from_pretrained(mid)
        _GEMMA_MODEL=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.bfloat16,device_map="auto")
    return _GEMMA_MODEL,_GEMMA_PROC

def _load_moondream():
    global _MD_MODEL
    if _MD_MODEL is None:
        dev=_device()
        if dev=="cuda": torch.set_default_device("cuda")
        _MD_MODEL=AutoModelForCausalLM.from_pretrained("moondream/moondream3-preview",trust_remote_code=True,
            dtype=torch.bfloat16 if dev=="cuda" else torch.float32,device_map={"": "cuda"} if dev=="cuda" else None)
        try:_MD_MODEL.compile()
        except: pass
    return _MD_MODEL

def _load_sam2_t():
    global _SAM2_T_MODEL,_SAM2_T_PROC
    if _SAM2_T_MODEL is None:
        dev=_device()
        _SAM2_T_MODEL=Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(dev)
        _SAM2_T_PROC=Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    return _SAM2_T_MODEL,_SAM2_T_PROC

def dense_caption(image_path,max_new_tokens=512):
    model,proc=_load_gemma()
    img=Image.open(image_path).convert("RGB")
    img=_to_gemini_image(img,768)
    msgs=[{"role":"system","content":REGION_DENSE_CAPTION_META_PROMPT},
          {"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"Generate the region-level dense caption now."}]}]
    prompt=proc.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False)
    inputs=proc(images=[img],text=prompt,return_tensors="pt").to(model.device)
    with torch.inference_mode():
        ids=model.generate(**inputs,max_new_tokens=max_new_tokens)
    return proc.decode(ids[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True).strip()

def parse_dense_caption(txt):
    out=[]
    for ln in txt.splitlines():
        m=re.match(r"\s*\[(\d+)\s*:\s*(.+?)\]\s*$",ln)
        if m: out.append((int(m.group(1)),m.group(2)))
    return sorted(out,key=lambda x:x[0])

def md_boxes(img,text):
    md=_load_moondream(); W,H=img.size
    res=md.detect(img,text); objs=res.get("objects") or []
    b=[]
    for o in objs:
        x0=int(max(0,min(W,(o.get("x_min",o.get("xmin"))*W))))
        y0=int(max(0,min(H,(o.get("y_min",o.get("ymin"))*H))))
        x1=int(max(0,min(W,(o.get("x_max",o.get("xmax"))*W))))
        y1=int(max(0,min(H,(o.get("y_max",o.get("ymax"))*H))))
        x0,x1=sorted([x0,x1]); y0,y1=sorted([y0,y1])
        if (x1-x0)>=2 and (y1-y0)>=2: b.append([x0,y0,x1,y1])
    return b

def sam2_segment_boxes(img,boxes):
    if not boxes: return None
    model,proc=_load_sam2_t()
    dev=next(model.parameters()).device
    inp=proc(images=img,input_boxes=[boxes],return_tensors="pt").to(dev)
    with torch.no_grad(): out=model(**inp,multimask_output=False)
    masks=proc.post_process_masks(out.pred_masks,inp["original_sizes"])[0]
    m=masks.detach().cpu().numpy()
    if m.ndim==2: m=m[None,...]
    merged=(m>0.5).any(axis=0)
    H,W=np.array(img).shape[:2]
    return _ensure_hw(merged,H,W)

def build_auto_mask_generator(cfg,ckpt,device=None,pps=32,nms=0.7):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2/configs/sam2_1',version_base='1.2')
    sam2=build_sam2(cfg,ckpt,device=device or _device(),apply_postprocessing=False)
    return SAM2AutomaticMaskGenerator(sam2,points_per_side=pps,box_nms_thresh=nms,output_mode="binary_mask")

def exhaustive_masks(img_np,gen,min_pixels=1000):
    return [m["segmentation"].astype(bool) for m in gen.generate(img_np) if np.count_nonzero(m["segmentation"])>=min_pixels]

def mask_iou(a,b):
    a=a.astype(bool); b=b.astype(bool)
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
    return 0.0 if union==0 else inter/union

def refine_with_pool(region,pool,thr=0.5,used=None):
    used=used or set(); best_iou,best_idx=0.0,-1
    for i,m in enumerate(pool):
        if i in used: continue
        iou=mask_iou(region,m)
        if iou>best_iou: best_iou,best_idx=iou,i
    return (pool[best_idx],best_iou,best_idx) if best_iou>=thr and best_idx>=0 else (region,best_iou,-1)

def overlay_multi(img,masks,alpha=0.45,with_boxes=False,dim=0.0):
    base=img.convert("RGBA"); W,H=base.size
    arr=np.zeros((H,W,4),dtype=np.float32); arr[...,3]=dim*255
    import colorsys
    for i,m in enumerate(masks):
        sl=_ensure_hw(m,H,W)
        r,g,b=colorsys.hsv_to_rgb((i*0.618)%1.0,0.95,1.0)
        arr[sl]=[r*255,g*255,b*255,alpha*255]
    ov=Image.fromarray(np.clip(arr,0,255).astype(np.uint8),"RGBA")
    out=Image.alpha_composite(base,ov).convert("RGB")
    if with_boxes:
        for m in masks:
            box=_mask_bbox(_ensure_hw(m,H,W))
            if box: out=_draw_bbox(out,box)
    return out


def dense_caption_using_gemini(image_path,max_new_tokens=8192):
    cfg=types.GenerateContentConfig(system_instruction=REGION_DENSE_CAPTION_META_PROMPT,temperature=0.7)
    img=Image.open(image_path).convert("RGB")
    img=_to_gemini_image(img,768)
    for i in range(3):
        r=_GEMINI_CLIENT.models.generate_content(
            model=_GEMINI_MODEL,
            contents=[img, "Generate the region-level dense caption now.  Strictly follow the format: [<index>: <short region caption>] for each line."],
            config=cfg)
        _cost_log(r, "dense_caption")

        # --- NEW: early return if Gemini blocks for prohibited content ---
        pf = getattr(r, "prompt_feedback", None)
        if pf is not None:
            br = getattr(pf, "block_reason", None)
            # Be robust to enum/string/None
            br_name = getattr(br, "name", None) or str(br or "")
            if "PROHIBITED_CONTENT" in br_name or "OTHER" in br_name:
                print("[GEMINI] Blocked by safety (PROHIBITED_CONTENT). Returning '0:None'.")
                return "0:None"
        # -----------------------------------------------------------------

        if (txt := getattr(r, "text", None)) is not None:
            return txt.strip()
        if i < 2:
            print(f"[GEMINI] dense_caption_using_gemini: r.text is None, retrying... ({i+1}/3)")
            print(f"[GEMINI] Response object: {r}")
            print(f"[GEMINI] Response finishReason: {getattr(r, 'finishReason', None)}")
            print(f"[GEMINI] Response block_reason: {getattr(r, 'block_reason', None)}")
            sleep(10)
    print("[GEMINI] dense_caption_using_gemini: Failed to get text after 3 attempts, returning empty string.")
    return "0:None"

# --- UPDATED: Gemini verification returns (bool, description) ---
def _gemini_yes_no(prompt,orig,overlay):
    orig=_to_gemini_image(orig,768)
    overlay=_to_gemini_image(overlay,768)
    cfg=types.GenerateContentConfig(response_mime_type="application/json",temperature=0.7)
    r=_GEMINI_CLIENT.models.generate_content(
        model=_GEMINI_MODEL,
        contents=[orig,overlay,(
            f"Task: Strictly verify if the red mask + green bounding box corresponds to the given text prompt: {prompt}\n" 
            "Strictly check for correctness of the entity mentioned, its attributes, and its location in the image.\n"
            "If correct, also give a region-level description. For the description do not get biased by the prompt, just describe by solely focusing on the image content.\n"
            "Respond strictly as JSON: {\"output\": true|false, \"description\": \"...\"}."
        )],
        config=cfg)
    _cost_log(r, "verify")
    try:
        j=json.loads(r.text or "{}"); return bool(j.get("output")), j.get("description","")
    except: return False,""

def _gemini_pick_better(prompt,overlay_init,overlay_ref):
    overlay_init=_to_gemini_image(overlay_init,768)
    overlay_ref=_to_gemini_image(overlay_ref,768)
    cfg=types.GenerateContentConfig(response_mime_type="application/json",temperature=0.7)
    r=_GEMINI_CLIENT.models.generate_content(
        model=_GEMINI_MODEL,
        contents=[overlay_init,overlay_ref,(
            f"Strictly compare two segmentations (red mask + green bounding box) for the given text prompt: {prompt}\n" 
            "Pick the higher-quality mask (coverage, tight bounding box, fewer leaks/holes). If both are bad then output null.\n"
            "Answer strictly as JSON: {\"output\": \"initial\"|\"refined\"}.\n"
        )],
        config=cfg)
    _cost_log(r, "compare")
    try:return json.loads(r.text or "{}").get("output")
    except:return None

def _dense_caption_from_accepted_list(accepted: List[Dict]) -> str:
    """
    Build the region-level dense caption string:
    0: description
    1: description
    ...
    Uses 'description' if present; falls back to 'label'.
    """
    lines = []
    for a in sorted(accepted, key=lambda x: int(x.get("index", 0))):
        idx = int(a.get("index", 0))
        desc = (a.get("description") or "").strip()
        if not desc:
            desc = (a.get("label") or "").strip()
        lines.append(f"{idx}: {desc}")
    return "\n".join(lines)

def _gemini_generate_conversational_prompts(concept_meta_prompt: str, original_image_path: str, overlaid_image_path: str, dense_caption_str: str, out_dir: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Fill concept_meta_prompt with {DENSE_CAPTION}, call Gemini with the image + prompt,
    and save the resulting JSON next to other outputs.
    """
    meta = concept_meta_prompt.replace("{DENSE_CAPTION}", dense_caption_str)
    original_image = Image.open(original_image_path).convert("RGB")
    original_image = _to_gemini_image(original_image, 768)
    overlaid_image = Image.open(overlaid_image_path).convert("RGB")
    overlaid_image = _to_gemini_image(overlaid_image, 768)
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.7)

    try:
        resp = _GEMINI_CLIENT.models.generate_content(
            model=_GEMINI_MODEL,
            contents=[original_image, overlaid_image, meta],
            config=cfg
        )
        _cost_log(resp, "prompts")
        text = resp.text or "{}"
        try:
            obj = json.loads(text)
        except Exception:
            # Keep raw if parsing fails (rare)
            obj = {"raw": text}
        out_path = os.path.join(out_dir, "conversational_segmentation_prompts.json")
        with open(out_path, "w") as f:
            json.dump(obj, f, indent=2)
        return obj, out_path
    except Exception as e:
        # Non-fatal: just return None and let the pipeline proceed
        err_path = os.path.join(out_dir, "conversational_segmentation_prompts.error.txt")
        with open(err_path, "w") as f:
            f.write(str(e))
        return None, None

# --- helpers (place above run_pipeline) ---
def _get_font(size: int = 28):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()

def _wrap_text(draw, text, max_w, font):
    words=text.split(); lines=[]; cur=""
    for w in words:
        t=(cur+" "+w).strip()
        if draw.textlength(t,font=font)<=max_w: cur=t
        else: 
            if cur: lines.append(cur)
            cur=w
    if cur: lines.append(cur)
    return lines

def _viz_prompt(img, masks, prompt, out_path, alpha=0.45):
    W,H=img.size
    meas=Image.new("RGB",(10,10)); d=ImageDraw.Draw(meas)
    font=_get_font()
    lines=_wrap_text(d, prompt, W-32, font)
    th=sum(d.textbbox((0,0),ln,font=font)[3]-d.textbbox((0,0),ln,font=font)[1]+2 for ln in lines)+16
    over=overlay_multi(img,masks,alpha,with_boxes=True,dim=0.6) if masks else img.copy()
    out=Image.new("RGB",(W,H+th),(0,0,0)); out.paste(over,(0,th))
    D=ImageDraw.Draw(out); y=8
    for ln in lines:
        D.text((16,y),ln,fill=(255,255,255),font=font); y+=d.textbbox((0,0),ln,font=font)[3]-d.textbbox((0,0),ln,font=font)[1]+2
    out.save(out_path)
# --- end helpers ---

def preload_models(cfg=None, ckpt=None, points_per_side=32, nms_thresh=0.7):
    # one-time warmup to GPU
    # _load_gemma()
    _load_moondream()
    _load_sam2_t()
    if cfg and ckpt:
        return build_auto_mask_generator(cfg, ckpt, _device(), points_per_side, nms_thresh)

SEG_PROMPT_VERIFICATION_PROMPT = """
You are validating whether a mask correctly identifies what a referring expression describes in an image.

**Your Task:**
Given an image, a referring expression, and optionally a mask (shown by a bounding box), determine if the mask is correct.

**Rules:**

**If a mask IS present:**
Accept it ONLY if ALL of these are true:
1. The masked region corresponds to the target that the expression describes.
2. The mask includes NOTHING else beyond what the expression describes. The mask should capture the primary/most prominent instances that match the expression - if there are other unmasked regions that also match, accept the mask as long as it includes the most obvious or salient examples.
3. The expression is a reasonable way to refer to something in this image.

**If NO mask is present:**
Accept it ONLY if there is truly nothing in the image that matches the expression.

**Important:** Be generous with natural referring expressions. People describe things in various valid ways. Focus on whether the mask matches what was described, not whether the description is perfect.

---

Respond in JSON format:
{"accept": true|false, "reason": "<brief explanation>"}

**Expression to verify:** {prompt}
"""
def _gemini_verify_prompt_text(prompt, orig_path, viz_path):
    orig = Image.open(orig_path).convert("RGB")
    orig = _to_gemini_image(orig, 768)
    over = Image.open(viz_path).convert("RGB")
    over = _to_gemini_image(over, 768)
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.7)
    r = _GEMINI_CLIENT.models.generate_content(
        model=_GEMINI_MODEL,
        contents=[orig, over,
            (SEG_PROMPT_VERIFICATION_PROMPT.replace("{prompt}", prompt))
        ],
        config=cfg
    )
    _cost_log(r, "verify_prompt")
    try:
        j = json.loads(r.text or "{}")
        return bool(j.get("accept")), (j.get("reason") or "")
    except:
        return False, ""


def run_pipeline(image_path,out_dir,cfg,ckpt,gen,max_regions=None,iou_thresh=0.5,points_per_side=32,nms_thresh=0.7,min_pool_pixels=1000):
    timer=StepTimer()
    _cost_reset()
    os.makedirs(out_dir,exist_ok=True)
    with timer.track("load_image"):
        img=Image.open(image_path).convert("RGB")
        img=_to_gemini_image(img,768)
        img_np=np.array(img)
    with timer.track("dense_caption"):
        cap=dense_caption_using_gemini(image_path)
    with timer.track("parse_dense_caption"):
        regs=parse_dense_caption(cap)
    if max_regions: regs=regs[:max_regions]
    # with timer.track("build_auto_mask_generator"):
    #     gen=build_auto_mask_generator(cfg,ckpt,_device(),points_per_side,nms_thresh)
    with timer.track("exhaustive_masks"):
        pool=exhaustive_masks(img_np,gen,min_pool_pixels)

    accepted_dir=os.path.join(out_dir,"accepted_regions"); os.makedirs(accepted_dir,exist_ok=True)
    results=[]; used=set(); final_masks=[]; marks=[]; accepted=[]
    for idx,label in regs:
        with timer.track("region:md_boxes"):
            boxes=md_boxes(img,label)
        if boxes:
            with timer.track("region:sam2_segment_boxes"):
                init_mask=sam2_segment_boxes(img,boxes)
        else:
            init_mask=None
        ref_mask,iou,ref_idx=(init_mask,0.0,-1)
        if init_mask is not None and pool:
            with timer.track("region:refine_with_pool"):
                ref_mask,iou,ref_idx=refine_with_pool(init_mask,pool,iou_thresh,used)
            if ref_idx>=0: used.add(ref_idx)
        mark_pt=_mask_center(ref_mask) or _mask_center(init_mask) or _box_center(boxes[0] if boxes else None)
        init_overlay=ref_overlay=None
        init_overlay_without_mask=None
        ref_overlay_without_mask=None
        if init_mask is not None:
            with timer.track("region:overlay_initial"):
                im,_=_overlay_with_bbox(img,init_mask,(mark_pt[0],mark_pt[1],idx) if mark_pt else None)
                init_overlay=im
                # im.save(os.path.join(out_dir,f"{idx:03d}_{_sanitize(label)}_initial.png"))
                init_overlay_without_mask, _ = _overlay_with_bbox_without_mask(img, init_mask, (mark_pt[0], mark_pt[1], idx) if mark_pt else None)
                # init_overlay_without_mask.save(os.path.join(out_dir, f"{idx:03d}_{_sanitize(label)}_initial_no_mask.png"))
        if ref_mask is not None:
            with timer.track("region:overlay_refined"):
                im,_=_overlay_with_bbox(img,ref_mask,(mark_pt[0],mark_pt[1],idx) if mark_pt else None)
                ref_overlay=im 
                # im.save(os.path.join(out_dir,f"{idx:03d}_{_sanitize(label)}_refined.png"))
                ref_overlay_without_mask, _ = _overlay_with_bbox_without_mask(img, ref_mask, (mark_pt[0], mark_pt[1], idx) if mark_pt else None)
                # ref_overlay_without_mask.save(os.path.join(out_dir, f"{idx:03d}_{_sanitize(label)}_refined_no_mask.png"))
        ver_init,desc_init=(False,""); ver_ref,desc_ref=(False,"")
        if init_overlay is not None:
            with timer.track("region:gemini_verify_initial"):
                ver_init,desc_init=_gemini_yes_no(label,init_overlay_without_mask,init_overlay)
                ver_ref,desc_ref=ver_init,desc_init
        # if ref_overlay is not None:
        #     with timer.track("region:gemini_verify_refined"):
        #         ver_ref,desc_ref=_gemini_yes_no(label,img,ref_overlay)

        winner=None
        if ver_init:
            with timer.track("region:gemini_pick_better"):
                winner=_gemini_pick_better(label,init_overlay,ref_overlay)

        chosen=None; desc=""
        if ver_init and not ver_ref: 
            if winner=="initial":
                chosen,desc=init_mask,desc_init
        elif ver_ref and not ver_init: 
            if winner=="refined":
                chosen,desc=ref_mask,desc_ref
        elif ver_init and ver_ref:
            if winner=="initial": 
                chosen,desc=(init_mask,desc_init)
            else:
                chosen,desc=(ref_mask,desc_ref)
        else:
            chosen=None; desc=""

        if chosen is not None and ver_init:
            final_masks.append(chosen); marks.append((mark_pt,len(accepted)))
            out_path=os.path.join(accepted_dir,f"{len(accepted):03d}_{_sanitize(label)}.png")
            im,_=_overlay_with_bbox(img,chosen,(mark_pt[0],mark_pt[1],len(accepted)) if mark_pt else None)
            im.save(out_path)
            accepted.append({"index":len(accepted),"label":label,"description":desc,"path":out_path})

        results.append({"index":idx,"label":label,"verify_initial":ver_init,"desc_init":desc_init,"desc_ref":desc_init,"output":winner})

    # save all the accepted masks overlaid on the original image
    final_overlay_path=None
    if final_masks:
        comp=overlay_multi(img,final_masks,alpha=0.45)
        for pt,idx in marks: comp=_draw_mark(comp,pt,idx)
        final_overlay_path=os.path.join(out_dir,"all_regions_colored.png")
        comp.save(final_overlay_path)
        # also save a version without the masks
        img.save(os.path.join(out_dir,"original_image.png"))

    # save all the masks individually as well in format suitable for image segmentation tasks - image and also numpy arrays in the accepted directory
    final_masks_dir=None
    final_bboxes_dir=None
    if final_masks:
        final_masks_dir=os.path.join(accepted_dir,"masks")
        final_bboxes_dir=os.path.join(accepted_dir,"bboxes")
        os.makedirs(final_masks_dir,exist_ok=True)
        os.makedirs(final_bboxes_dir,exist_ok=True)
        for i,m in enumerate(final_masks):
            mimg=Image.fromarray((m.astype(np.uint8))*255); mimg.save(os.path.join(final_masks_dir,f"{i:03d}_mask.png"))
            np.savez_compressed(os.path.join(final_masks_dir,f"{i:03d}_mask.npz"),mask=m.astype(np.uint8))
            b=_mask_bbox(m)
            if b:
                with open(os.path.join(final_bboxes_dir,f"{i:03d}_bbox.txt"),"w") as f:
                    f.write(f"{b[0]} {b[1]} {b[2]} {b[3]}\n")
            # update accepted entry with mask and bbox
            if i<len(accepted):
                accepted[i]["bbox"]=b
                accepted[i]["mask_path"]=os.path.join(final_masks_dir,f"{i:03d}_mask.png")
                accepted[i]["mask_npz_path"]=os.path.join(final_masks_dir,f"{i:03d}_mask.npz")
    
    if accepted:
        with open(os.path.join(accepted_dir,"accepted.json"),"w") as f: json.dump(accepted,f,indent=2)

    # --- NEW: Build dense caption from accepted + generate conversational prompts with Gemini ---
    dense_caption_from_accepted = _dense_caption_from_accepted_list(accepted) if accepted else ""
    out_dir_orig = out_dir  # keep original out_dir for saving prompts
    for concept in ["ENTITIES", "SPATIAL", "AFFORDANCES", "RELATIONS", "PHYSICS"]:
        CONCEPT_META_PROMPT = CONCEPT_META_PROMPTS.get(concept)
        conversational_prompts = None
        conversational_prompts_path = None
        out_dir = os.path.join(out_dir_orig, concept.lower())
        os.makedirs(out_dir, exist_ok=True)

        if dense_caption_from_accepted:
            conversational_prompts, conversational_prompts_path = _gemini_generate_conversational_prompts(
                concept_meta_prompt=CONCEPT_META_PROMPT,
                original_image_path=image_path,
                overlaid_image_path=final_overlay_path,
                dense_caption_str=dense_caption_from_accepted,
                out_dir=out_dir
            )
        # --- END NEW ---

        # --- NEW: visualize generated prompts (satisfying masks only) ---
        viz_dir=None

        # print("Conversational prompts:", conversational_prompts)

        if isinstance(conversational_prompts, list):
            plist=conversational_prompts
            if plist:
                viz_dir=os.path.join(out_dir,"generated_prompts_viz"); os.makedirs(viz_dir,exist_ok=True)
                acc_dir = os.path.join(viz_dir, "ACCEPTED"); rej_dir = os.path.join(viz_dir, "REJECTED")
                os.makedirs(acc_dir, exist_ok=True); os.makedirs(rej_dir, exist_ok=True)
                prompt_verifs = []
                for i,p in enumerate(plist):
                    sats=[int(x) for x in (p.get("satisfying") or []) if x is not None and str(x).isdigit() and 0<=int(x)<len(final_masks)]
                    # print("satisfying masks for prompt:", i, p.get("prompt"), sats)
                    masks=[final_masks[j] for j in sats if j>=0 and j<len(final_masks)]
                    text=str(p.get("prompt") or "").strip()
                    name=f"{i:02d}_{_sanitize(p.get('sub_concept') or p.get('concept_family') or 'prompt')}.png"
                    viz_path = os.path.join(viz_dir, name)
                    _viz_prompt(img,masks,text,viz_path,alpha=0.45)
                    with timer.track("prompt:gemini_verify"):
                        ok, reason = _gemini_verify_prompt_text(text, image_path, viz_path)
                    tgt_dir = acc_dir if ok else rej_dir
                    tgt = os.path.join(tgt_dir, name)
                    try: shutil.copy2(viz_path, tgt)
                    except: tgt = viz_path
                    prompt_verifs.append({
                        "index": i, "prompt": text, "satisfying": sats,
                        "verdict": "ACCEPT" if ok else "REJECT",
                        "reason": reason,
                        "src_path": viz_path,
                        "classified_path": tgt
                    })
        # --- END NEW ---

        timing_summary=timer.report()
        # add total time
        total_time = sum(item["total_seconds"] for item in timing_summary)
        timing_summary.append({"step": "total", "total_seconds": total_time, "count": 1, "avg_seconds": total_time})

        summary={"image_path":image_path,
                "dense_caption":cap,
                "results":results,
                "accepted_count":len(accepted),
                "accepted_dir":accepted_dir,
                "final_overlay_path":final_overlay_path,
                "accepted_dense_caption":dense_caption_from_accepted,
                "conversational_prompts_path":conversational_prompts_path,
                "conversational_prompts":conversational_prompts,
                "generated_prompts_viz_dir":viz_dir,
                "timing":timing_summary,
                "gemini_usage":{"input_tokens":_COST["in"],
                                "output_tokens":_COST["out"],
                                "usd":round(_COST["usd"],6),
                                "calls":_COST["calls"]},
                "prompt_verifications": prompt_verifs if viz_dir else None}

        with open(os.path.join(out_dir,"summary.json"),"w") as f: 
            json.dump(summary,f,indent=2)

    print(f"[GEMINI] total calls={_COST['calls']} in={_COST['in']} out={_COST['out']}" + (f" cost=${_COST['usd']:.6f}" if _COST["usd"] else ""))
    # print total time
    print(f"[TIMER] total time={total_time:.2f} seconds")
    
    return summary


if __name__=="__main__":

    import sys
    import os
    from pathlib import Path
    import argparse

    argument_parser = argparse.ArgumentParser(description="Run the full image segmentation and conversational prompt generation pipeline.")
    argument_parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image or directory of images.")
    argument_parser.add_argument("--config", "-c", type=str, default="sam2.1_hiera_l.yaml", help="Path to SAM2 config YAML file.")
    argument_parser.add_argument("--checkpoint", "-k", type=str, default="/data/aadarsh/code/simulation/working-dir/grind/sam2/checkpoints/sam2.1_hiera_large.pt", help="Path to SAM2 checkpoint file.")
    argument_parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save outputs.")

    args = argument_parser.parse_args()

    inp  = args.input
    cfg  = args.config
    ckpt = args.checkpoint
    out_dir = args.output_dir

    gen = preload_models(cfg, ckpt, points_per_side=64, nms_thresh=0.7)

    p = Path(inp)
    exts = ("*.jpg","*.jpeg","*.png","*.webp","*.bmp","*.tif","*.tiff")
    imgs = [p] if p.is_file() else [q for pat in exts for q in sorted(p.glob(pat))]

    for i, im in enumerate(imgs):
        print(f"\n\n=== [{i+1}/{len(imgs)}] Processing image: {im} ===")
        out = os.path.join(out_dir, f"out_region_refine_{im.stem}")
        # first check if it exists and has summary.json for all concepts
        all_done = True
        for concept in ["entities", "spatial", "affordances", "relations", "physics"]:
            summary_path = os.path.join(out, concept, "summary.json")
            if not os.path.exists(summary_path):
                all_done = False
                break
        if all_done:
            print(f"Output already exists for all concepts at: {out}, skipping...")
            continue
        s = run_pipeline(str(im), out, cfg, ckpt, gen, max_regions=8, iou_thresh=0.5,
                         points_per_side=32, nms_thresh=0.7, min_pool_pixels=1000)
        print("Saved to:", out)
