#!/usr/bin/env python3
"""
Export DataEngine runs into train/eval dataset formats used by ConvSeg.
"""

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


CONCEPTS = ["entities", "spatial", "affordances", "relations", "physics"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export DataEngine run outputs to train.py JSONL and/or eval.py JSON payload. "
            "Reads concept summaries and builds merged masks from accepted prompt regions."
        )
    )
    parser.add_argument(
        "--runs_root",
        required=True,
        type=str,
        help=(
            "Root directory containing run folders (for example: out_region_refine_*). "
            "Can also point directly to a single run folder."
        ),
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Directory where exported dataset files and staged images/masks are written.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "both"],
        default="both",
        help="Which output(s) to write: train JSONL, eval JSON, or both.",
    )
    parser.add_argument(
        "--path_mode",
        choices=["relative", "absolute"],
        default="relative",
        help="Path style for image/mask references inside dataset files.",
    )
    return parser.parse_args()


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned[:120] if cleaned else "item"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _read_mask_bool(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im.convert("L"), dtype=np.uint8)
    return arr > 0


def _save_mask(path: Path, mask_bool: np.ndarray) -> None:
    out = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    out.save(path)


def _is_run_dir(path: Path) -> bool:
    return any((path / concept / "summary.json").is_file() for concept in CONCEPTS)


def _discover_run_dirs(runs_root: Path) -> List[Path]:
    found: List[Path] = []
    if _is_run_dir(runs_root):
        found.append(runs_root)
    for child in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if _is_run_dir(child):
            found.append(child)
    uniq = []
    seen = set()
    for path in found:
        rp = path.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def _resolve_existing_path(raw_path: Optional[str], run_dir: Path) -> Optional[Path]:
    if not raw_path:
        return None
    raw = Path(raw_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(run_dir / raw)
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def _parse_satisfying(raw) -> List[int]:
    out: List[int] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if idx >= 0:
            out.append(idx)
    return sorted(set(out))


def _resolve_mask_path(run_dir: Path, mask_by_idx: Dict[int, Path], idx: int) -> Optional[Path]:
    path = mask_by_idx.get(idx)
    if path and path.is_file():
        return path.resolve()
    fallback = run_dir / "accepted_regions" / "masks" / f"{idx:03d}_mask.png"
    if fallback.is_file():
        return fallback.resolve()
    return None


def _path_for_dataset(path: Path, out_dir: Path, path_mode: str) -> str:
    path = path.resolve()
    if path_mode == "absolute":
        return str(path)
    return str(path.relative_to(out_dir.resolve()))


def _stage_image(
    src_image: Path,
    run_name: str,
    images_dir: Path,
    image_cache: Dict[str, Path],
) -> Path:
    src_key = str(src_image.resolve())
    if src_key in image_cache:
        return image_cache[src_key]

    suffix = src_image.suffix.lower() or ".png"
    stem = _safe_name(src_image.stem)
    base_name = f"{_safe_name(run_name)}__{stem}{suffix}"
    dest = images_dir / base_name
    counter = 1
    while dest.exists():
        dest = images_dir / f"{_safe_name(run_name)}__{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(src_image, dest)
    image_cache[src_key] = dest.resolve()
    return image_cache[src_key]


def _load_accepted_masks(run_dir: Path, skipped: Counter) -> Tuple[List[dict], Dict[int, Path]]:
    accepted_json = run_dir / "accepted_regions" / "accepted.json"
    if not accepted_json.is_file():
        skipped["missing_accepted_json"] += 1
        return [], {}

    try:
        accepted = _load_json(accepted_json)
    except Exception:
        skipped["invalid_accepted_json"] += 1
        return [], {}

    if not isinstance(accepted, list):
        skipped["invalid_accepted_json_type"] += 1
        return [], {}

    mask_by_idx: Dict[int, Path] = {}
    for item in accepted:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue

        mask_path = _resolve_existing_path(item.get("mask_path"), run_dir)
        if mask_path is None:
            fallback = run_dir / "accepted_regions" / "masks" / f"{idx:03d}_mask.png"
            if fallback.is_file():
                mask_path = fallback.resolve()
        if mask_path is not None:
            mask_by_idx[idx] = mask_path

    return accepted, mask_by_idx


def _iter_concept_summaries(run_dir: Path) -> Iterable[Tuple[str, Path]]:
    for concept in CONCEPTS:
        summary_path = run_dir / concept / "summary.json"
        if summary_path.is_file():
            yield concept, summary_path


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(runs_root)
    if not run_dirs:
        raise RuntimeError(
            f"No run directories found under {runs_root}. Expected concept summaries under: {CONCEPTS}."
        )

    skipped = Counter()
    train_rows: List[dict] = []
    eval_items: List[dict] = []
    per_run_stats: List[dict] = []
    image_cache: Dict[str, Path] = {}
    sample_counter = 0

    for run_dir in run_dirs:
        run_name = run_dir.name
        run_exported = 0
        run_skipped_before = sum(skipped.values())

        _, mask_by_idx = _load_accepted_masks(run_dir, skipped)
        if not mask_by_idx:
            per_run_stats.append(
                {
                    "run_dir": str(run_dir),
                    "exported_samples": 0,
                    "skipped_in_run": sum(skipped.values()) - run_skipped_before,
                }
            )
            continue

        for concept, summary_path in _iter_concept_summaries(run_dir):
            try:
                summary = _load_json(summary_path)
            except Exception:
                skipped["invalid_summary_json"] += 1
                continue

            prompt_verifications = summary.get("prompt_verifications")
            if not isinstance(prompt_verifications, list):
                skipped["missing_prompt_verifications"] += 1
                continue

            image_path = _resolve_existing_path(summary.get("image_path"), run_dir)
            if image_path is None:
                skipped["missing_image_path"] += 1
                continue

            for prompt_idx, prompt_info in enumerate(prompt_verifications):
                if not isinstance(prompt_info, dict):
                    skipped["invalid_prompt_verification_item"] += 1
                    continue

                verdict = str(prompt_info.get("verdict", "")).strip().upper()
                if verdict != "ACCEPT":
                    skipped["verdict_not_accept"] += 1
                    continue

                prompt = str(prompt_info.get("prompt", "")).strip()
                if not prompt:
                    skipped["empty_prompt"] += 1
                    continue

                satisfying = _parse_satisfying(prompt_info.get("satisfying"))
                if not satisfying:
                    skipped["empty_satisfying"] += 1
                    continue

                mask_paths: List[Path] = []
                missing_indices: List[int] = []
                for idx in satisfying:
                    mask_path = _resolve_mask_path(run_dir, mask_by_idx, idx)
                    if mask_path is None:
                        missing_indices.append(idx)
                    else:
                        mask_paths.append(mask_path)
                if missing_indices:
                    skipped["missing_satisfying_mask"] += 1
                    continue

                try:
                    merged_mask: Optional[np.ndarray] = None
                    for mask_path in mask_paths:
                        current = _read_mask_bool(mask_path)
                        if merged_mask is None:
                            merged_mask = current
                        else:
                            if current.shape != merged_mask.shape:
                                raise ValueError(
                                    f"Mask shape mismatch: {current.shape} vs {merged_mask.shape}"
                                )
                            merged_mask = np.logical_or(merged_mask, current)
                    if merged_mask is None:
                        skipped["empty_merged_mask"] += 1
                        continue
                except Exception:
                    skipped["failed_mask_merge"] += 1
                    continue

                staged_image = _stage_image(image_path, run_name, images_dir, image_cache)
                sample_counter += 1
                sample_id = (
                    f"{_safe_name(run_name)}_{concept}_{prompt_idx:03d}_{sample_counter:06d}"
                )
                staged_mask = masks_dir / f"{sample_id}.png"
                _save_mask(staged_mask, merged_mask)

                image_ref = _path_for_dataset(staged_image, out_dir, args.path_mode)
                mask_ref = _path_for_dataset(staged_mask.resolve(), out_dir, args.path_mode)

                train_rows.append(
                    {
                        "image": image_ref,
                        "mask_merged": mask_ref,
                        "prompt": prompt,
                    }
                )
                eval_items.append(
                    {
                        "id": sample_id,
                        "image": image_ref,
                        "mask": mask_ref,
                        "prompt": prompt,
                        "concept": concept,
                    }
                )
                run_exported += 1

        per_run_stats.append(
            {
                "run_dir": str(run_dir),
                "exported_samples": run_exported,
                "skipped_in_run": sum(skipped.values()) - run_skipped_before,
            }
        )

    written_files: Dict[str, str] = {}

    if args.mode in ("train", "both"):
        dataset_jsonl = out_dir / "dataset.jsonl"
        with dataset_jsonl.open("w", encoding="utf-8") as f:
            for row in train_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written_files["dataset_jsonl"] = str(dataset_jsonl)

    if args.mode in ("eval", "both"):
        items_json = out_dir / "items.json"
        payload = {
            "dataset": out_dir.name,
            "count": len(eval_items),
            "items": eval_items,
        }
        _write_json(items_json, payload)
        written_files["items_json"] = str(items_json)

    manifest = {
        "runs_root": str(runs_root),
        "out_dir": str(out_dir),
        "mode": args.mode,
        "path_mode": args.path_mode,
        "concepts": CONCEPTS,
        "discovered_run_dirs": [str(p) for p in run_dirs],
        "counts": {
            "train_rows": len(train_rows),
            "eval_items": len(eval_items),
            "copied_images": len(image_cache),
            "written_masks": sample_counter,
        },
        "skipped_reason_counts": dict(skipped),
        "per_run": per_run_stats,
        "written_files": written_files,
    }
    manifest_path = out_dir / "export_manifest.json"
    _write_json(manifest_path, manifest)

    print(f"Discovered run dirs: {len(run_dirs)}")
    print(f"Exported samples: {sample_counter}")
    print(f"Copied images: {len(image_cache)}")
    print(f"Wrote manifest: {manifest_path}")
    if written_files:
        for key, value in written_files.items():
            print(f"Wrote {key}: {value}")


if __name__ == "__main__":
    main()

