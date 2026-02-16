# Conversational Image Segmentation

This repository contains the official code for the paper [**Conversational Image Segmentation: Grounding Abstract Concepts with Scalable Supervision**](https://glab-caltech.github.io/ConverSeg/).  
**Authors:** [Aadarsh Sahoo](https://aadsah.github.io/), [Georgia Gkioxari](https://gkioxari.github.io/)

<p align="center">
  <a href="https://glab-caltech.github.io/converseg/">
    <img src="https://img.shields.io/badge/Project%20Page-Website-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Project Page"/>
  </a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper"/>
  </a>
  <a href="https://huggingface.co/datasets/aadarsh99/ConverSeg">
    <img src="https://img.shields.io/badge/Dataset-Hugging%20Face-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Dataset"/>
  </a>
  <a href="https://huggingface.co/spaces/aadarsh99/ConverSeg">
    <img src="https://img.shields.io/badge/Space-Hugging%20Face-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Space"/>
  </a>
</p>

<p align="center">
  <img src="docs/teaser.png" alt="ConverSeg teaser" width="700"/>
</p>

***



## Table of Contents

1. [Setup](#1-setup)
2. [Checkpoints](#2-checkpoints)
3. [Dataset Formats](#3-dataset-formats)
4. [Training](#4-training)
5. [Evaluation](#5-evaluation)
6. [DataEngine Quickstart](#6-dataengine-quickstart)
7. [Outputs](#7-outputs)
8. [Citation](#8-citation)
9. [Acknowledgments](#9-acknowledgments)

***

## 1. Setup

Use the provided conda environment file:

```bash
git clone --recurse-submodules https://github.com/AadSah/ConverSeg.git
cd ConverSeg
conda env create -f converseg.yml
conda activate converseg
```

***

## 2. Checkpoints

Download the released checkpoints from Hugging Face:

- https://huggingface.co/aadarsh99/ConverSeg-Net-3B

These are raw checkpoint files for this repository (for example `.torch` and LoRA adapter files), not Hugging Face `from_pretrained` model format.

```bash
git lfs install
git clone https://huggingface.co/aadarsh99/ConverSeg-Net-3B ./checkpoints/ConverSeg-Net-3B
```

Run inference with `demo.py` by pointing to the downloaded checkpoint paths:

```bash
python demo.py \
  --final_ckpt ./checkpoints/ConverSeg-Net-3B/ConverSeg-Net_sam2_90000.torch.torch \
  --plm_ckpt ./checkpoints/ConverSeg-Net-3B/ConverSeg-Net_plm_90000.torch \
  --lora_ckpt ./checkpoints/ConverSeg-Net-3B/lora_plm_adapter_90000
  --model_cfg sam2_hiera_l.yaml \
  --base_ckpt /path/to/sam2_hiera_large.pt \
  --image /path/to/image.jpg \
  --prompt "the left-most person" \
  --device cuda \
  --out_dir ./demo_outputs
```

You can also run `demo.py` in interactive mode by omitting `--image` and `--prompt`.

***

## 3. Dataset Formats

### Training format (`dataset.jsonl`)

`train.py` reads a JSONL file (default: `dataset.jsonl`) inside `--data-dir`.

Each line:

- `image`: image path (relative to `--data-dir` or absolute)
- `mask_merged`: segmentation mask path (relative or absolute)
- `prompt`: optional text prompt

Example:

```json
{"image":"images/0001.jpg","mask_merged":"masks/0001.png","prompt":"the left-most person"}
```

Minimal layout:

```text
my_data/
  dataset.jsonl
  images/
  masks/
```

### Evaluation JSON format (`items.json`)

When using `--input_json`, `eval.py` expects:

```json
{
  "dataset": "chunk_01",
  "count": 2,
  "items": [
    {"id":"0001","image":"images/0001.jpg","mask":"masks/0001.png","prompt":"the left-most person"},
    {"id":"0002","image":"images/0002.jpg","mask":"masks/0002.png","prompt":"a red object near the chair"}
  ]
}
```
## 📋 TODO
- [ ] Release training data.

***

## 4. Training

```bash
python train.py \
  --data-dir /path/to/my_data \
  --dataset-jsonl dataset.jsonl \
  --model-cfg sam2_hiera_s.yaml \
  --checkpoint /path/to/sam2_hiera_small.pt \
  --device cuda \
  --steps 3000 \
  --batch-size 4 \
  --acc 4 \
  --lr 1e-4 \
  --wd 1e-4 \
  --out ./ckpts \
  --name ConverSeg_sam2
```

Notes:

- Checkpoints and LoRA adapters are written under `--out`.
- TensorBoard logs are written to `--out/tb/<name>`.

***

## 5. Evaluation

`eval.py` supports two modes.

### Hugging Face mode (default)

```bash
python eval.py \
  --final_ckpt ./ckpts/ConverSeg_sam2_final.torch \
  --plm_ckpt ./ckpts/ConverSeg_sam2_plm_final.torch \
  --lora_ckpt ./ckpts/lora_plm_adapter_final \
  --model_cfg sam2_hiera_l.yaml \
  --base_ckpt /path/to/sam2_hiera_large.pt \
  --hf_dataset aadarsh99/ConverSeg \
  --hf_config default \
  --hf_splits sam_seeded,human_annotated \
  --device cuda \
  --save_preds ./preds_hf
```

### JSON mode

```bash
python eval.py \
  --input_json /path/to/items.json \
  --final_ckpt ./ckpts/ConverSeg_sam2_final.torch \
  --plm_ckpt ./ckpts/ConverSeg_sam2_plm_final.torch \
  --lora_ckpt ./ckpts/lora_plm_adapter_final \
  --model_cfg sam2_hiera_l.yaml \
  --base_ckpt /path/to/sam2_hiera_large.pt \
  --device cuda \
  --save_preds ./preds_json
```

***

## 6. DataEngine Quickstart

Generate conversational supervision from raw images, then export into ConverSeg train/eval formats.

Install extra dependency:

```bash
pip install google-genai
```

Set environment variable:

```bash
export GOOGLE_API_KEY=<your_key>
```

Run generation:

```bash
python dataengine/run.py \
  --input /path/to/image_or_dir \
  --config sam2.1_hiera_l.yaml \
  --checkpoint /path/to/sam2.1_hiera_large.pt \
  --output_dir /path/to/dataengine_runs
```

Export for training/evaluation:

```bash
python dataengine/tools/export_dataset.py \
  --runs_root /path/to/dataengine_runs \
  --out_dir /path/to/ConverSeg_export \
  --mode both \
  --path_mode relative
```

See `dataengine/DATAENGINE.md` for full schemas and failure modes.

***

## 7. Outputs

From `train.py` (`--out`):

- `<name>_<step>.torch`: SAM2 checkpoints
- `<name>_plm_<step>.torch`: language adapter checkpoints
- `lora_plm_adapter_<step>/`: LoRA adapter snapshots
- `tb/<name>/`: TensorBoard logs
- `val/step_<step>.png`: validation panels

From `eval.py` (`--save_preds`):

- `*_pred.png`: predicted masks
- `*_gt.png`: GT masks
- `*_orig.png`: source images
- `*_panel.png`: overlays
- `*_prompt.txt`: prompt text

***

## 8. Citation

```bibtex
@article{sahoo202XConverSeg,
  title   = {Conversational Image Segmentation: Grounding Abstract Concepts with Scalable Supervision},
  author  = {Sahoo, Aadarsh and Gkioxari, Georgia},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {202X}
}
```

For any questions or issues, please open a GitHub issue or contact [Aadarsh](mailto:aadarsh.sahoo.99@gmail.com). Thank you for your interest in our work!


***

## 9. Acknowledgments

ConverSeg builds on [SAM2](https://github.com/facebookresearch/sam2).
