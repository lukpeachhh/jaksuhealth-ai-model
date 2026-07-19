# JaksuHealth AI Model

Deep-learning models for multi-class segmentation of AMD-related lesions in
Optical Coherence Tomography (OCT) images.

> **Repository scope:** JaksuHealth was developed by a team as a complete
> software project. This repository contains only the AI model pipeline that I
> was responsible for: dataset preparation, OCT preprocessing, semantic
> segmentation model development, training, evaluation, inference, and model
> integration support. The frontend, backend, database, and other application
> components are not included.

## Achievement

JaksuHealth won **1st place in the regional round of Startup Thailand League**
and advanced to the national round.

## My contribution

- Prepared and validated OCT images and segmentation masks.
- Developed the retinal region-of-interest preprocessing pipeline.
- Implemented and compared U-Net, U-Net++, and DeepLabV3+.
- Trained the models under the same experimental conditions.
- Evaluated overall and per-lesion segmentation performance.
- Prepared an inference pipeline for integration into the JaksuHealth software.

## Lesion classes

The mask label mapping is defined once in
[`src/jaksuhealth_ai/constants.py`](src/jaksuhealth_ai/constants.py).

| Class ID | Label | Description |
|---:|---|---|
| 0 | Background | Non-lesion pixels |
| 1 | SRF | Subretinal fluid |
| 2 | IRF | Intraretinal fluid |
| 3 | PED | Pigment epithelial detachment |
| 4 | SHRM | Subretinal hyperreflective material |
| 5 | IS/OS | Inner segment / outer segment abnormality |

> **Mapping verification:** The processed integer masks used by this project
> use `2 = IRF` and `3 = PED`. The mapping was checked against representative
> masks and the AMD-SD publication's color description, where green represents
> IRF and blue represents PED. The Kaggle data card may show a different class
> order, so this repository always uses the mapping defined in `constants.py`
> for training, evaluation, visualization, and software integration.

## Compared architectures

- U-Net
- U-Net++
- DeepLabV3+

The default configurations use the same encoder, pretrained initialization,
input size, preprocessing, augmentation policy, loss, optimizer, scheduler,
random seed, and data split. Only the segmentation architecture changes.

## Repository structure

```text
jaksuhealth-ai-model/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
├── src/jaksuhealth_ai/
├── scripts/
├── notebooks/
│   └── 00_kaggle_run_all.ipynb
├── results/
├── docs/
└── tests/
```

## Dataset

This project uses the **AMD-SD** OCT dataset. The primary dataset reference is:

- Hu, Y., Gao, Y., Gao, W. *et al.* **AMD-SD: An Optical Coherence Tomography
  Image Dataset for wet AMD Lesions Segmentation.** *Scientific Data* 11, 1014
  (2024).
- Article and dataset documentation:
  <https://www.nature.com/articles/s41597-024-03844-6#Sec6>
- DOI: <https://doi.org/10.1038/s41597-024-03844-6>

The publication reports 3,049 OCT B-scan images from 138 patients and 156 eyes,
with expert annotations for SRF, IRF, PED, SHRM, and IS/OS. This repository does
not redistribute images, masks, demographic data, or other AMD-SD files.

The experiments in this repository use the processed Kaggle copy identified as:

```text
gaoweihao/amd-sd
```

The Kaggle copy is used only as the runtime data source. The scientific paper
above is the primary reference and should be cited when presenting or
publishing results.

### Citation

```bibtex
@article{hu2024amdsd,
  title   = {AMD-SD: An Optical Coherence Tomography Image Dataset for wet AMD Lesions Segmentation},
  author  = {Hu, Yunwei and Gao, Yundi and Gao, Weihao and others},
  journal = {Scientific Data},
  volume  = {11},
  pages   = {1014},
  year    = {2024},
  doi     = {10.1038/s41597-024-03844-6}
}
```

## Dataset layout

After attaching or extracting the processed Kaggle dataset, the expected layout
is:

```text
AMD-SD/
├── images/
│   ├── 100_1.png
│   └── ...
└── masks/
    ├── 100_1.png
    └── ...
```

The preparation script extracts a group ID from the filename prefix before the
first underscore. In AMD-SD filenames such as `n_x.png`, the publication defines
`n` as the **eye ID**. Therefore, the default split is an **eye-level grouped
split**, not a patient-level split. Images from the same eye are kept in one
split.

A true patient-level split requires the eye-to-patient mapping from the original
demographic metadata. Do not describe the default Kaggle-only split as
patient-level unless that mapping has been applied.

## Run on Kaggle from GitHub

The recommended entry point is:

```text
notebooks/00_kaggle_run_all.ipynb
```

### Kaggle setup

1. Upload or import `00_kaggle_run_all.ipynb` into Kaggle.
2. Attach the AMD-SD dataset using **Add Input**.
3. Enable a GPU accelerator.
4. Enable Internet access for cloning GitHub and installing dependencies.
5. Set `GITHUB_USERNAME` in the notebook.
6. Choose `RUN_MODE = "single"` or `RUN_MODE = "all"`.
7. Run the notebook from top to bottom.

Kaggle Inputs are read-only. Generated splits, checkpoints, metrics, and figures
are written under `/kaggle/working/jaksuhealth-ai-model`.

The notebook is safe to rerun in the same session: it updates the cloned
repository without deleting ignored `data/` or `results/` files, validates the
GPU, locates AMD-SD automatically, verifies image-mask pairs and label IDs,
checks the canonical class mapping, prepares the grouped split with
`--overwrite`, and then executes exactly one selected run mode.

### Manual Kaggle commands

Clone the repository:

```python
GITHUB_USERNAME = "YOUR_GITHUB_USERNAME"
REPOSITORY = "jaksuhealth-ai-model"

!git clone https://github.com/{GITHUB_USERNAME}/{REPOSITORY}.git
%cd /kaggle/working/{REPOSITORY}
!pip install -q -r requirements.txt
```

Prepare the dataset:

```python
!python scripts/prepare_dataset.py \
    --data-dir "{DATA_DIR}" \
    --output-dir data/AMD-SD_Split \
    --seed 42 \
    --overwrite
```

Train and evaluate one model:

```python
!python scripts/train.py --config configs/unet.yaml --device cuda
!python scripts/evaluate.py \
    --config configs/unet.yaml \
    --checkpoint results/runs/unet/best_model.pth \
    --split test \
    --device cuda
```

Run all architectures sequentially:

```python
!python scripts/run_all_experiments.py --device cuda
```

Training all three models may exceed one Kaggle session or GPU quota. For final
experiments, running one architecture per saved Kaggle version is safer.

## Local installation

Python 3.9 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

The scripts add `src/` to `PYTHONPATH` automatically, so they can be run from
the repository root without packaging the project first.

## Prepare the grouped split locally

```bash
python scripts/prepare_dataset.py \
  --data-dir data/AMD-SD \
  --output-dir data/AMD-SD_Split \
  --seed 42
```

Generated layout:

```text
data/AMD-SD_Split/
├── train/images
├── train/masks
├── val/images
├── val/masks
├── test/images
├── test/masks
├── split_groups.json
└── split_summary.csv
```

## Train models

```bash
python scripts/train.py --config configs/unet.yaml
python scripts/train.py --config configs/unetplusplus.yaml
python scripts/train.py --config configs/deeplabv3plus.yaml
```

Each run creates a directory under `results/runs/` containing the checkpoint,
configuration, training history, curves, and validation metrics.

## Evaluate a checkpoint

```bash
python scripts/evaluate.py \
  --config configs/unet.yaml \
  --checkpoint results/runs/unet/best_model.pth \
  --split test
```

The evaluator exports overall metrics, per-class metrics, a confusion matrix,
and sample prediction figures.

## Compare the models

```bash
python scripts/compare_models.py \
  --runs-dir results/runs \
  --output-dir results
```

Generated files include:

- `results/model_comparison.csv`
- `results/per_class_metrics.csv`
- `results/figures/model_comparison.png`

## Predict one OCT image

```bash
python scripts/predict.py \
  --config configs/unet.yaml \
  --checkpoint results/runs/unet/best_model.pth \
  --image path/to/oct_image.png \
  --output-dir results/prediction
```

## Testing

```bash
PYTHONPATH=src pytest -q
```

## Fair-comparison protocol

For a meaningful architecture comparison, keep the following fixed:

- grouped train/validation/test split;
- encoder and pretrained initialization;
- input resolution and ROI crop;
- data augmentation;
- loss function and class weights;
- optimizer, learning rate, batch size, and epochs;
- early-stopping criterion;
- random seed;
- test-time augmentation setting.

Use validation results for model selection. Use the test split only for the
final report.

## Result reporting

Pixel accuracy alone can be misleading because background pixels may dominate
an OCT mask. Report at least:

- macro Dice/F1 excluding background;
- macro IoU excluding background;
- precision and recall;
- per-class Dice and IoU;
- parameter count;
- inference time per image.

The CSV files committed under `results/` contain headers only. Replace them with
results produced from real checkpoints before presenting the repository as a
completed model comparison.

## Software integration

The model receives a grayscale OCT B-scan and returns a multi-class lesion mask.
See [`docs/software_integration.md`](docs/software_integration.md) for the input
and output contract used by the software team.

## Limitations

- Performance may change across OCT devices, hospitals, acquisition protocols,
  and patient populations.
- Small or rare lesion classes may have unstable metrics.
- The repository does not include external clinical validation.
- Results depend on the exact dataset version, mask conversion, and split files.

## Medical disclaimer

This repository is a research and portfolio project. It is **not a medical
device** and must not be used as a replacement for diagnosis or clinical
judgment by qualified healthcare professionals.

## License and ownership

The source code in this repository is released under the
[MIT License](LICENSE). The MIT License applies only to code that the repository
owners have the right to publish. It does **not** apply to AMD-SD images, masks,
demographic files, the Scientific Data article, model weights with separate
terms, or third-party libraries.

Before making the repository public, confirm that the model code can be shared
and that relevant JaksuHealth teammates, supervisors, competition organizers,
and the university do not hold conflicting intellectual-property rights.
Student projects can still use a software license; the important question is
who owns the code and whether the owners agree to release it.
