# FruitLite-L2

A compact PyTorch CNN for classifying **apple, banana, orange, strawberry and
pineapple** at 224×224 and 32×32. This Deep Learning midterm project includes
training, automatic checkpoint recovery, image prediction and the measured
results of one completed experiment. Both models were trained from scratch.

| Input | Test accuracy | Correct / total | Macro F1 | Final epochs | Weights |
|---|---:|---:|---:|---:|---|
| [224×224](results/mid224/test/metrics.json) | **100.00%** | 250 / 250 | 1.000000 | 148 | Raw |
| [32×32](results/mid32/test/metrics.json) | **99.60%** | 249 / 250 | 0.996000 | 129 | EMA |

These are results on this project's fixed 250-image test set. They do not
establish accuracy on unseen sources or verified independent physical fruit.
Source rights, full visual labels and near-duplicate review remain incomplete;
22,685 training images lack verified physical groups. Dataset images and
checkpoint binaries are kept locally and are **not distributed** by this repo.

See [test metrics](results/test_summary.csv), [reproduction details](results/reproduction.json)
and [source review](results/dataset/source_review.json).

## Model

FruitLite-L2 uses residual blocks with parallel 3×3 and 5×5 depthwise
convolutions, channel attention, global average pooling and a five-class head.
The same graph is used at both input sizes.

| Component | Output channels | Blocks | First stride |
|---|---:|---:|---:|
| Stem | 16 | 1 | 2 |
| Stage 1 | 32 | 1 | 1 |
| Stage 2 | 56 | 2 | 2 |
| Stage 3 | 96 | 3 | 2 |
| Stage 4 | 240 | 4 | 2 |
| Head | 384 → 5 | 1 | 1 |

**1,250,879 learnable parameters**, **860,413,737 FLOPs at 224** and
**17,973,033 FLOPs at 32**, batch size 1. The profiler counts 2 × MACs plus
normalization, activation, pooling, gating and residual additions. Both models
satisfy the assignment's ≤6M parameter and <1G FLOP limits.

## Install

Use an existing **Python 3.13 base installation**, PyTorch 2.7.1 and torchvision
0.22.1. The Windows entry points verify the selected Microsoft Store Python;
no virtual environment is created. Preserve a working CUDA installation.

```powershell
$py = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\python3.13.exe'
& $py -m pip install --only-binary=:all: --no-build-isolation -r requirements.txt
```

CPU execution is supported. The recorded experiment used an RTX 4060 Laptop
GPU with PyTorch 2.7.1+cu118. The dependency pins also accept that CUDA build.

## Data

Each resolution has 25,000 training and 250 test images: 5,000 train and 50
test per class. The assignment says *approximately* 5,000 training images per
class; exactly 5,000 was the implementation target.

| Dataset | Encoding | Image bytes | Limit |
|---|---|---:|---:|
| Mid224 | RGB AVIF, 224×224 | 23,148,133 | 25,000,000 |
| Mid32 | RGB WebP, 32×32 | 14,948,274 | 20,000,000 |

Restore the exact images through your own authorized copy of the dataset.
The [manifest](results/dataset/manifest.csv) records paths, source identities,
review flags and SHA256 hashes; it contains no images. This repository does
not offer an automatic dataset download or claim redistribution rights.

```text
DataNew/
  Mid224/{train,test}/{class}/{id}.avif
  Mid32/{train,test}/{class}/{id}.webp
data/metadata/
  splits.csv
  build_settings.json
```

On a fresh checkout, restore the two metadata inputs from the published
evidence after supplying `DataNew/`:

```powershell
New-Item -ItemType Directory -Force data/metadata
Copy-Item results/dataset/manifest.csv data/metadata/splits.csv
Copy-Item results/dataset/build_settings.json data/metadata/build_settings.json
```

The input audit checks quotas, format, geometry, image hashes, duplicates and
dataset capacity. A decoded audit is cached; subsequent starts still verify
the inventory and SHA256 of every encoded image. `--full-audit` forces decoding.

## Train

Run one file from the repository root:

```powershell
& $py -X utf8 -B train.py
```

The pipeline creates a fixed 22,500 / 2,500 development split, trains Mid224
for 180 epochs and Mid32 for 220 epochs, selects epoch budgets and raw/EMA
weights by validation, then refits each model on all 25,000 training images.
Test is evaluated only after both training plans finish. No pretrained weights
are loaded; test accuracy never selects the checkpoint.

The recorded run selected 148 epochs with raw weights at 224, and 129 epochs
with EMA weights at 32. Full histories and the frozen development partition
are in [results/](results/).

AdamW uses LR 0.0015, weight decay 0.01, five warmup epochs and cosine decay.
The recipe includes label smoothing, mixup, random crops, flips, rotation,
color jitter, random erasing, AMP and EMA. See [configs/](configs/) and the
exported settings for every parameter. Training accuracy uses augmented
images and primary labels during mixup; validation and test do not.

**Stop and resume:** press Ctrl+C, then run the same command. `last.pt` restores
model, optimizer, AMP scaler, EMA, RNG and history at the last completed epoch.
An interrupted epoch repeats. The console prints one summary per completed
epoch; batch and per-class details go to files.

Each phase overwrites `last.pt` and `best.pt`. In development, `best` is selected
by validation accuracy, then loss. During the full-pool refit, it follows the
previously locked weight type and final epoch. Other files include
`settings.json`, `epochs.csv`, `epochs.json`, `console.log`, `batches.jsonl`
and learning curves. Outputs are local under `outputs/{run-name}/`.

An already completed run is verified and displayed without retraining or
repeating test inference. Use a new name to start another experiment:

```powershell
& $py -X utf8 -B train.py --run-name experiment_02
& $py -X utf8 -B train.py --check --run-name input_check
```

Changing source code, configuration or data cannot silently resume an
incompatible partial run. The recorded experiment's original source snapshot
remains beside its local checkpoints, and its original code commit is recorded
in [reproduction.json](results/reproduction.json).

## Evaluate and predict

With the local completed checkpoints present, verify the saved test receipts,
checkpoint hashes, prediction probabilities and recomputed classification
metrics without reading test images or changing artifacts:

```powershell
& $py -X utf8 -B evaluate.py
```

Classify an image using either trained resolution:

```powershell
& $py -X utf8 -B evaluate.py --image path/to/fruit.jpg --size 224
& $py -X utf8 -B evaluate.py --image path/to/fruit.jpg --size 32 --device cpu
```

Prediction converts to RGB, resizes to the selected input size and applies
the training normalization. Images outside the five classes still receive a
five-class prediction; this is not an unknown-object detector.

## Repository

```text
train.py, evaluate.py   Supported entry points
src/                   Model, loaders, audit, training, metrics and inference
configs/               Training configurations
tests/                 Data-free fixtures for training, resume and evaluation
results/               Histories, settings, test predictions, plots and provenance
```

`DataNew/`, `data/` and `outputs/` are ignored local artifacts. Original image
collection scripts, preparation caches and preliminary reports have been
removed. The course document is retained locally under `data/reference/`.

## Development

```powershell
& $py -m pip install --only-binary=:all: --no-build-isolation -r requirements-dev.txt
& $py -X utf8 -B -m pytest -q -p no:cacheprovider
& $py -m ruff check .
```

The tests create their own tiny datasets; they do not require private images,
download models or train the full experiment. They cover model limits,
deterministic splits, hash tampering, completed-epoch checkpoint transactions,
resume consistency, AMP overflow and saved-result verification. The GitHub
workflow checks public-tree hygiene; the full Python suite is verified locally.

Contributions should include focused tests, preserve the locked-test protocol
and avoid committing datasets, checkpoints or credentials. See the
[security policy](.github/SECURITY.md) for reporting sensitive issues.

## License

Project code is licensed under [Apache-2.0](LICENSE).
Copyright 2026 FruitLite-L2 contributors. The license does not grant rights to
third-party images or the course assignment document.
