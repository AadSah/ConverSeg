# DataEngine

DataEngine generates conversational segmentation supervision from raw images, then exports it into the formats expected by `train.py` and `eval.py`.

## Directory Structure

```text
dataengine/
  DATAENGINE.md
  run.py
  prompts/
    __init__.py
    concept_specific_meta_prompts.py
    concept_specific_meta_prompts_for_negatives.py
    region_dense_caption_meta_prompt.py
  meta_prompts/  # backward-compatible shims
    __init__.py
    concept_specific_meta_prompts.py
    concept_specific_meta_prompts_for_negatives.py
    region_dense_caption_meta_prompt.py
  tools/
    export_dataset.py
```

`dataengine/prompts/*` is the canonical prompt source namespace.  
`dataengine/meta_prompts/*` remains available for legacy imports.

## Prerequisites

From repo root:

```bash
pip install -r requirements.txt
pip install -e ./sam2
pip install google-genai
```

Environment:

- `GOOGLE_API_KEY` (required by Gemini calls in `dataengine/run.py`)
- GPU recommended (SAM2 + VLM path is heavy)

Run all commands from repo root (`ConvSeg/`) unless noted.

## Quickstart: Generate DataEngine Runs

Single image:

```bash
python dataengine/run.py \
  --input /path/to/image.jpg \
  --config sam2.1_hiera_l.yaml \
  --checkpoint /path/to/sam2.1_hiera_large.pt \
  --output_dir /path/to/dataengine_runs
```

Directory of images:

```bash
python dataengine/run.py \
  --input /path/to/image_dir \
  --config sam2.1_hiera_l.yaml \
  --checkpoint /path/to/sam2.1_hiera_large.pt \
  --output_dir /path/to/dataengine_runs
```

The run script writes one folder per input image:

```text
/path/to/dataengine_runs/
  out_region_refine_<image_stem>/
```

## Run Output Contract

Per image run (`out_region_refine_<image_stem>/`):

```text
out_region_refine_<image_stem>/
  original_image.png
  all_regions_colored.png
  accepted_regions/
    accepted.json
    000_<label>.png
    ...
    masks/
      000_mask.png
      000_mask.npz
      ...
    bboxes/
      000_bbox.txt
      ...
  entities/
    conversational_segmentation_prompts.json
    generated_prompts_viz/
      ACCEPTED/
      REJECTED/
    summary.json
  spatial/
    summary.json
  affordances/
    summary.json
  relations/
    summary.json
  physics/
    summary.json
```

## Export to `train.py` / `eval.py` Formats

Use `dataengine/tools/export_dataset.py` to convert run outputs.

### Export both train and eval formats

```bash
python dataengine/tools/export_dataset.py \
  --runs_root /path/to/dataengine_runs \
  --out_dir /path/to/convseg_export \
  --mode both \
  --path_mode relative
```

### Export train-only

```bash
python dataengine/tools/export_dataset.py \
  --runs_root /path/to/dataengine_runs \
  --out_dir /path/to/convseg_export \
  --mode train \
  --path_mode relative
```

### Export eval-only

```bash
python dataengine/tools/export_dataset.py \
  --runs_root /path/to/dataengine_runs \
  --out_dir /path/to/convseg_export \
  --mode eval \
  --path_mode relative
```

Exporter output:

```text
/path/to/convseg_export/
  images/
  masks/
  dataset.jsonl           # if mode=train|both
  items.json              # if mode=eval|both
  export_manifest.json    # always
```

### Export filtering logic

Exporter includes only prompts where:

- `summary.json -> prompt_verifications[*].verdict == "ACCEPT"`
- `summary.json -> prompt_verifications[*].satisfying` is non-empty
- all satisfying mask indices resolve to existing accepted masks

For each exported prompt, mask is the boolean union of all satisfying masks.

## Direct Handoff to Training

```bash
python train.py \
  --data-dir /path/to/convseg_export \
  --dataset-jsonl dataset.jsonl \
  --model-cfg sam2_hiera_s.yaml \
  --checkpoint /path/to/sam2_hiera_small.pt \
  --device cuda
```

`train.py` expects each JSONL line to provide:

- `image`
- `mask_merged`
- `prompt`

## Direct Handoff to Evaluation

```bash
python eval.py \
  --input_json /path/to/convseg_export/items.json \
  --final_ckpt /path/to/final_model.torch \
  --plm_ckpt /path/to/final_plm.torch \
  --model_cfg sam2_hiera_l.yaml \
  --base_ckpt /path/to/sam2_hiera_large.pt \
  --device cuda
```

`eval.py --input_json` expects:

- top-level: `dataset`, `count`, `items`
- each item: `id`, `image`, `mask`, `prompt`

## Field-Level Schemas

### `accepted_regions/accepted.json`

Type: `List[Object]`

Required fields used downstream:

- `index` (`int`)
- `label` (`str`)
- `path` (`str`) overlay path
- `bbox` (`[x0, y0, x1, y1]`) if available
- `mask_path` (`str`) PNG mask path
- `mask_npz_path` (`str`) NPZ mask path

### `<concept>/summary.json`

Important fields:

- `image_path` (`str`)
- `accepted_dir` (`str`)
- `conversational_prompts` (`list|dict|null`)
- `prompt_verifications` (`list|null`)

`prompt_verifications[*]` shape:

- `index` (`int`)
- `prompt` (`str`)
- `satisfying` (`List[int]`)
- `verdict` (`"ACCEPT"` or `"REJECT"`)
- `reason` (`str`)
- `src_path` (`str`)
- `classified_path` (`str`)

### Exported `dataset.jsonl`

One JSON object per line:

```json
{"image":"images/<file>.jpg","mask_merged":"masks/<file>.png","prompt":"<text prompt>"}
```

### Exported `items.json`

```json
{
  "dataset": "convseg_export",
  "count": 2,
  "items": [
    {"id":"sample_1","image":"images/a.jpg","mask":"masks/a.png","prompt":"segment ...","concept":"entities"},
    {"id":"sample_2","image":"images/b.jpg","mask":"masks/b.png","prompt":"segment ...","concept":"spatial"}
  ]
}
```


