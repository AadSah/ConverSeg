import argparse
import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


from models.language_adapter import LanguageAdapter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.util_functions import (
    bin_mask,
    erode,
    iou,
    overlay_mask,
    put_text,
    read_img_mask,
    seed,
    split,
    to_bgr_u8,
)


def load_samples_from_jsonl(data_dir, jsonl="dataset.jsonl", num_samples=None):
    path = os.path.join(data_dir, jsonl)
    out = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            try:
                ip = os.path.join(data_dir, item["image"])
                mp = os.path.join(data_dir, item["mask_merged"])
                pr = item.get("prompt", "")
                if os.path.isfile(ip) and os.path.isfile(mp):
                    out.append((ip, mp, pr))
            except Exception as e:
                print(f"Re-trying reading line: {line}\n{e}")
                try:
                    ip = os.path.join(item["image"])
                    mp = os.path.join(item["mask_merged"])
                    pr = item.get("prompt", "")
                    if os.path.isfile(ip) and os.path.isfile(mp):
                        out.append((ip, mp, pr))
                except Exception as e2:
                    raise RuntimeError(f"Error reading line: {line}\n{e2}")
    if not out:
        raise RuntimeError("No samples found. Check paths.")
    print("Loaded", len(out), "samples from", path)
    return out[:num_samples] if num_samples else out

class PairListDataset(Dataset):
    def __init__(self, pairs): self.pairs = list(pairs)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]

def collate_keep_list(batch): return batch

def dice_loss_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)           # [B,H,W]
    inter = torch.sum(probs * target, (1,2))
    denom = torch.sum(probs, (1,2)) + torch.sum(target, (1,2))
    return (1. - ((2*inter + eps) / (denom + eps))).mean()


def _select_val_subset(val_pairs, max_dim, k):
    """Pick up to k non-empty GT samples; fallback to first k if none found."""
    picks = []
    for (ip, mp, pr) in val_pairs:
        try:
            _, ann = read_img_mask(ip, mp, max_dim)
            picks.append((ip, mp, pr))
        except Exception as e:
            print(f"  skipped due to error: {e}")
            continue
        if len(picks) >= k: break
    if not picks:
        picks = val_pairs[:k]
    return picks


def _validate_once(
    predictor, plm, val_subset, image_pe_const, dec_dev, dec_dtype,
    max_dim, er_k, er_it, save_dir, step
):
    was_md = predictor.model.sam_mask_decoder.training
    was_pe = predictor.model.sam_prompt_encoder.training
    was_plm = plm.training
    predictor.model.sam_mask_decoder.eval()
    predictor.model.sam_prompt_encoder.eval()
    plm.eval()
    panels, ious = [], []
    with torch.no_grad():
        for (img_p, msk_p, pr) in val_subset:
            img, ann = read_img_mask(img_p, msk_p, max_dim)
            gt_full = erode(bin_mask(ann), er_k, er_it).astype(np.float32)

            predictor.set_image(img)
            img_emb = predictor._features["image_embed"][-1].unsqueeze(0)
            hi = [lvl[-1].unsqueeze(0) for lvl in predictor._features["high_res_feats"]]
            _, C, Hm, Wm = img_emb.shape

            prompt = pr or "segment"
            sp, dp = plm([prompt], Hm, Wm, image_paths=[img_p])
            sp, dp = sp.to(dec_dev, dec_dtype), dp.to(dec_dev, dec_dtype)
            img_emb = img_emb.to(dec_dev, dec_dtype)
            hi = [h.to(dec_dev, dec_dtype) for h in hi]
            image_pe = image_pe_const  # keep batch dim == 1

            low, scores, *_ = predictor.model.sam_mask_decoder(
                image_embeddings=img_emb,
                image_pe=image_pe,
                sparse_prompt_embeddings=sp,
                dense_prompt_embeddings=dp,
                multimask_output=False,
                repeat_image=False,
                high_res_features=hi,
            )

            best = scores.argmax(dim=1).item()
            prob = torch.sigmoid(low[0, best]).detach()
            h, w = prob.shape[-2], prob.shape[-1]

            gt = torch.from_numpy(cv2.resize(gt_full, (w, h), interpolation=cv2.INTER_NEAREST)).to(prob.device)
            pred = (prob > 0.5).float()
            iou_val = iou(pred.unsqueeze(0), gt.unsqueeze(0)).item()
            ious.append(iou_val)

            disp_w = 512
            disp_h = int(disp_w * img.shape[0] / max(1, img.shape[1]))
            base   = to_bgr_u8(cv2.resize(img, (disp_w, disp_h)))

            gt_vis = cv2.resize(gt.cpu().numpy().astype(np.float32), (disp_w, disp_h), cv2.INTER_NEAREST) > 0.5
            pr_vis = cv2.resize(pred.cpu().numpy().astype(np.float32), (disp_w, disp_h), cv2.INTER_NEAREST) > 0.5

            left  = base.copy(); put_text(left, f"Prompt: {prompt[:80]}  |  IoU {iou_val:.3f}")
            mid   = overlay_mask(base, gt_vis, color=(0,255,0), alpha=0.65, edge_px=1); put_text(mid, "GT")
            right = overlay_mask(base, pr_vis, color=(0,0,255), alpha=0.65, edge_px=1); put_text(right, f"Pred  |  {prompt[:60]}", y=48)

            panels.append(cv2.hconcat([left, mid, right]))

        if panels:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f"step_{step}.png"), cv2.vconcat(panels))
        mean_iou = float(np.mean(ious)) if ious else 0.0
        print(f"[val] step {step}: mean IoU over {len(ious)} samples = {mean_iou:.4f}")
    
    predictor.model.sam_mask_decoder.train(was_md)
    predictor.model.sam_prompt_encoder.train(was_pe)
    plm.train(was_plm)

    return mean_iou


def train_loop(
    train_pairs,
    val_pairs,
    predictor: SAM2ImagePredictor,
    plm: LanguageAdapter,
    steps=3000,
    lr=1e-4,
    wd=1e-4,
    batch_size=4,
    acc=4,
    out="./ckpts",
    name="fine_tuned_sam2_batched",
    max_dim=1024,
    er_k=5,
    er_it=1,
    log_every=20,
    save_every=500,
    workers=4,
    seed_val=1337,
    val_every=200,
    val_count=6,
    val_dir="./ckpts/val",
    min_lr=1e-6,
    warmup_updates=-1,
    ):

    model = predictor.model
    model.train(); plm.train()

    for p in model.parameters(): p.requires_grad = False
    for p in model.sam_mask_decoder.parameters(): p.requires_grad = True
    for p in model.sam_prompt_encoder.parameters(): p.requires_grad = True

    # print the number of trainable parameters for each part
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    print("Trainable parameters (in M):")
    print(f"  SAM Mask Decoder: {count_params(model.sam_mask_decoder) / 1e6:.2f} {count_params(model.sam_mask_decoder)}")
    print(f"  SAM Prompt Encoder: {count_params(model.sam_prompt_encoder) / 1e6:.2f} {count_params(model.sam_prompt_encoder)}")
    print(f"  PLM Adapter: {count_params(plm) / 1e6:.2f} {count_params(plm)}")
    print(f"  PLM Backbone: {count_params(plm.backbone) / 1e6:.2f} {count_params(plm.backbone)}")

    # TensorBoard writer
    tb_dir = os.path.join(out, "tb", name)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    params = list(model.sam_mask_decoder.parameters()) + \
             list(model.sam_prompt_encoder.parameters()) + \
             list(plm.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    total_updates = max(1, math.ceil(steps / max(1, acc)))
    if warmup_updates < 0:
        warmup_updates = max(0, int(0.03 * total_updates))
    cos_updates = max(1, total_updates - warmup_updates)
    warmup_start_factor = 1e-8  

    if warmup_updates > 0:
        warmup = LinearLR(opt, start_factor=warmup_start_factor, total_iters=warmup_updates)
        cosine = CosineAnnealingLR(opt, T_max=cos_updates, eta_min=min_lr)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_updates])
    else:
        scheduler = CosineAnnealingLR(opt, T_max=cos_updates, eta_min=min_lr)

    dec_dev  = next(model.sam_mask_decoder.parameters()).device
    dec_dtype = next(model.sam_mask_decoder.parameters()).dtype
    image_pe_const = model.sam_prompt_encoder.get_dense_pe().to(dec_dev, dec_dtype)

    ds = PairListDataset(train_pairs)
    g = torch.Generator(); g.manual_seed(seed_val)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                        collate_fn=collate_keep_list, drop_last=False,
                        generator=g, persistent_workers=(workers>0))

    def infinite_loader():
        while True:
            for b in loader:
                yield b

    data_it = infinite_loader()
    os.makedirs(out, exist_ok=True); os.makedirs(val_dir, exist_ok=True)

    print("Preparing validation subset...")
    val_subset = _select_val_subset(val_pairs, max_dim, max(4, min(8, val_count)))
    train_subset = _select_val_subset(train_pairs, max_dim, max(4, min(8, val_count)))


    opt.zero_grad(set_to_none=True)
    update_idx = grad_acc = 0

    print("Starting training loop...")
    for step in range(1, steps+1):
        batch = next(data_it)  # list of (img_p, msk_p, prompt)

        imgs, gts, prompts, img_paths = [], [], [], []

        for (img_p, msk_p, pr) in batch:
            img, ann = read_img_mask(img_p, msk_p, max_dim)
            bmask = erode(bin_mask(ann), er_k, er_it)
            imgs.append(img)
            gts.append(bmask.astype(np.float32))
            prompts.append(pr or "segment")
            img_paths.append(img_p)

        if not imgs:
            continue

        # >>> Batched image encoding (no grads) <<<
        predictor.set_image_batch(imgs)
        img_emb_b = predictor._features["image_embed"].to(dec_dev, dec_dtype)        # [B, C, Hm, Wm]
        hi_b      = [lvl.to(dec_dev, dec_dtype) for lvl in predictor._features["high_res_feats"]]  # list[[B, Ck, Hk, Wk]]
        B, C, Hm, Wm = img_emb_b.shape

        sp, dp = plm(prompts, Hm, Wm, image_paths=img_paths)
        sp, dp = sp.to(dec_dev, dec_dtype), dp.to(dec_dev, dec_dtype)

        image_pe = image_pe_const  
        low, scores, *_ = model.sam_mask_decoder(
            image_embeddings=img_emb_b,
            image_pe=image_pe,
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dp,
            multimask_output=False,
            repeat_image=False,
            high_res_features=hi_b,
        )  

        logits = low[:, 0]
        h, w = logits.shape[-2:]
        gt_t = torch.stack([
            torch.from_numpy(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))
            for m in gts
        ], dim=0).to(dec_dev)

        bce = F.binary_cross_entropy_with_logits(logits, gt_t)
        dl  = 0.25 * dice_loss_from_logits(logits, gt_t)
        loss = (bce + dl) / acc
        loss.backward()
        grad_acc += 1

        if grad_acc == acc:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            grad_acc = 0; update_idx += 1
            scheduler.step()

        if (step % log_every) == 0:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                j = iou(preds, gt_t).item()
            print(f"step {step} (upd {update_idx}): loss~{(bce+dl).item():.4f} BCE~{bce.item():.4f} Dice~{dl.item():.4f} IoU~{j:.4f} (eff_batch={batch_size*acc})")
            # TensorBoard scalars
            writer.add_scalar("train/loss", (bce+dl).item(), step)
            writer.add_scalar("train/bce", bce.item(), step)
            writer.add_scalar("train/dice", dl.item(), step)
            writer.add_scalar("train/iou", j, step)
            writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)

        if val_every and (step % val_every) == 0:
            val_miou = _validate_once(predictor, plm, val_subset, image_pe_const, dec_dev, dec_dtype,
                           max_dim, er_k, er_it, val_dir, step)
            train_miou = _validate_once(predictor, plm, train_subset, image_pe_const, dec_dev, dec_dtype,
                           max_dim, er_k, er_it, val_dir.replace("val", "train"), step)
            writer.add_scalar("val/mean_iou", val_miou, step)
            writer.add_scalar("train_subset/mean_iou", train_miou, step)

        if save_every and (step % save_every) == 0:
            torch.save({"model": model.state_dict()}, os.path.join(out, f"{name}_{step}.torch"))
            torch.save({"plm": plm.state_dict()},   os.path.join(out, f"{name}_plm_{step}.torch"))
            plm.save_lora(os.path.join(out, f"lora_plm_adapter_{step}"))
            print(f"Saved checkpoints at step {step}")

    torch.save({"model": model.state_dict()}, os.path.join(out, f"{name}_final.torch"))
    torch.save({"plm": plm.state_dict()},   os.path.join(out, f"{name}_plm_final.torch"))
    plm.save_lora(os.path.join(out, "lora_plm_adapter_final"))
    print("Training complete.")

    val_full = _select_val_subset(val_pairs, max_dim, k=len(val_pairs))
    final_miou = _validate_once(
        predictor, plm, val_full, image_pe_const, dec_dev, dec_dtype,
        max_dim, er_k, er_it, val_dir, step="final"
    )
    writer.add_scalar("val/final_mean_iou", final_miou, steps)

    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-cfg", default="sam2_hiera_s.yaml")
    ap.add_argument("--checkpoint", default="sam2_hiera_small.pt")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--acc", type=int, default=4)
    ap.add_argument("--out", default="./ckpts")
    ap.add_argument("--name", default="fine_tuned_sam2_batched")
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--er-k", type=int, default=5)
    ap.add_argument("--er-it", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-samples", type=int, default=None)

    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--val-count", type=int, default=6)
    ap.add_argument("--val-dir", default="./ckpts/val")
    ap.add_argument("--dataset-jsonl", default="dataset.jsonl")
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-updates", type=int, default=-1, help="Warmup measured in optimizer updates; -1 => 3% of total updates")

    ap.add_argument("--resume-sam-path", default=None)
    ap.add_argument("--resume-plm-path", default=None)
    ap.add_argument("--resume-lora-path", default=None)

    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)

    ap.add_argument("--plm-model-name", nargs="?", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Name of the PLM model to use as backbone for the language adapter")

    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    seed(args.seed)

    pairs = load_samples_from_jsonl(args.data_dir, args.dataset_jsonl, num_samples=args.num_samples)
    train_pairs, val_pairs = split(pairs, 0.01, args.seed)

    print(f"Training samples: {len(train_pairs)}, Validation samples: {len(val_pairs)}")

    predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, args.checkpoint, device=args.device))
    C = predictor.model.sam_mask_decoder.transformer_dim
    print(f"Decoder transformer dim: {C}")

    plm = LanguageAdapter(
        model_name=args.plm_model_name,
        transformer_dim=C,
        n_sparse_tokens=0,
        use_dense_bias=True,
        use_lora=True, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        dtype=torch.bfloat16,
        device=args.device,
        use_image_input=True,
    ).to(args.device)

    print(plm)
    print(f"PLM Adapter total params: {sum(p.numel() for p in plm.parameters()) / 1e6:.2f} M")

    trainable = sum(p.numel() for p in plm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in plm.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if args.resume_sam_path:
        sd = torch.load(args.resume_sam_path, map_location="cpu")
        predictor.model.load_state_dict(sd["model"], strict=True)
        print(f"Resumed SAM model from {args.resume_sam_path}")
    if args.resume_plm_path:
        sd = torch.load(args.resume_plm_path, map_location="cpu")
        plm.load_state_dict(sd["plm"], strict=True)
        print(f"Resumed PLM adapter from {args.resume_plm_path}")
    if args.resume_lora_path and plm.peft_enabled:
        plm.load_lora(args.resume_lora_path)
        print(f"Resumed PLM LoRA weights from {args.resume_lora_path}")

    

    train_loop(
        train_pairs, val_pairs, predictor, plm,
        steps=args.steps, lr=args.lr, wd=args.wd,
        batch_size=args.batch_size, acc=args.acc,
        out=args.out, name=args.name, max_dim=args.max_dim,
        er_k=args.er_k, er_it=args.er_it, log_every=args.log_every,
        save_every=args.save_every, workers=args.workers, seed_val=args.seed,
        val_every=args.val_every, val_count=args.val_count, val_dir=args.val_dir,
        min_lr=args.min_lr, warmup_updates=args.warmup_updates,
    )

if __name__ == "__main__":
    main()
