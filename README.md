# Waste Instance Classification

Classify individual waste objects cropped from COCO bounding boxes with an AlexNet variant and two decision-tree baselines. The project includes instance loading, training, validation, evaluation and confusion-matrix plots.

## Models

| Entry | Model | Input |
| --- | --- | --- |
| `train.py`, `evaluate.py` | AlexNet variant trained from scratch | RGB crops resized to 192 × 192, scaled to [0, 1] |
| `decision_tree.py` | Decision tree on flattened pixels | The same 192 × 192 crops |
| `hog_baseline.py` | Per-channel HOG → standardization → PCA → decision tree | RGB crops resized to 128 × 128 |

The loader uses supplied instance boxes. The task is crop classification; box coordinates and category annotations are inputs. Category count and display names come from the COCO file. The original coursework used seven categories.

## Setup

Use Python 3.10 or later with a compatible PyTorch installation.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data format

Place your own COCO annotations at `data/coco/annotations.json` and keep each image path relative to `data/coco/`:

```text
data/coco/
  annotations.json
  JPEGImages/
    example.jpg
```

The annotations need `images`, `annotations` and `categories` arrays. For example, one instance can be represented as:

```json
{
  "images": [{"id": 17, "file_name": "JPEGImages/example.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 25, "image_id": 17, "category_id": 3, "bbox": [50, 60, 120, 90], "area": 10800, "iscrowd": 0}],
  "categories": [{"id": 3, "name": "paper"}]
}
```

This fragment illustrates the schema. Training needs at least seven annotated images to populate all three partitions. Bounding boxes use `[x, y, width, height]`. The loader converts images to RGB and maps sorted category IDs to consecutive labels starting at zero, so category IDs can be non-contiguous.

`data_split.py` assigns annotated images to 70% training, 15% validation and the remaining test partition using seed 42. Every crop from one image stays in the same partition. All entry points share this partition rule and category mapping. Keep the annotations file fixed between training and evaluation; changing its image or category list changes the split or label interpretation.

## Run

From the repository root:

```sh
# AlexNet: select an epoch count for the current dataset.
WASTE_EPOCHS=50 python train.py
python evaluate.py

# Fit and evaluate a decision tree on flattened crop pixels.
python decision_tree.py

# Fit the HOG/PCA baseline, then evaluate its saved model.
python hog_baseline.py train
python hog_baseline.py evaluate
```

The AlexNet entry retains the original defaults: Adam at 0.001, batch size 32, CPU execution and 1,000 epochs. It saves the model with the best validation accuracy to `outputs/best_model.pth`. The HOG entry fits its scaler, PCA and classifier on training crops and saves `outputs/hog_pca_tree.pkl`. Evaluate that pickle only from a locally trusted training run. The pixel-tree entry holds flattened crop arrays in memory during fitting.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `WASTE_DATA_ROOT` | `data/coco/` relative to the repository | Image root |
| `WASTE_ANNOTATIONS` | `annotations.json` under the image root | COCO annotation file |
| `WASTE_OUTPUT_DIR` | Repository `outputs/` | Weights, baseline model and plots |
| `WASTE_DEVICE` | `cpu` | PyTorch device, such as `cpu`, `cuda` or `mps` |
| `WASTE_BATCH_SIZE` | `32` | Batch size |
| `WASTE_EPOCHS` | `1000` | AlexNet training epochs |

Evaluation prints accuracy, a classification report and a confusion matrix. Confusion-matrix rows represent actual classes and columns represent predicted classes; plotted rows are normalized with empty-class rows kept at zero.

## Source and publication

The repository combines the author's `AI Project2` AlexNet and pixel-tree code with the `AI Lab4` HOG/PCA model. Publication changes replace machine-specific paths, share image-level splits, map category IDs, handle RGB input consistently, repair confusion-matrix axes, create output directories and replace the conflicting dependency list. Model structures and training objectives follow the selected coursework implementations.

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records the course framework and torchvision architecture provenance. The torchvision BSD license is retained for the model structure. Repository-wide reuse terms for the coursework code remain unspecified.

Publication preparation used source review. Local training, evaluation and tests were not run.
