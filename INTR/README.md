# INTR (ICLR 2024) - Extended Project README

This repository is based on INTR: A Simple Interpretable Transformer for Fine-Grained Image Classification and Analysis and includes additional engineering changes made in this project for practical training, resume, visualization, and single-image demo workflows.

Paper: https://arxiv.org/pdf/2311.04157.pdf

## 1. What This Repository Contains

Core INTR components:
- Transformer-based interpretable classifier with class queries.
- Training and evaluation entrypoint via main.py.
- Attention visualization tools.

Project extensions added here:
- K queries per class support (k as hyperparameter).
- Query aggregation modes (max/mean/sum).
- Resume-friendly k-query finetuning script.
- Single image demo script for prediction + heatmap generation.
- Dataset conversion utility for CUB_200_2011 to ImageFolder format.
- OOM-safe evaluation controls for quick subset experiments.

## 2. Environment and Dependencies

Recommended runtime:
- Python 3.8 (64-bit)
- PyTorch with CUDA support

Install dependencies:

```bash
pip install -r requirements.txt
```

Additional packages used in this project session:

```bash
pip install opencv-python seaborn scipy packaging
```

## 3. Dataset Layout

Expected dataset layout for training/eval:

```text
datasets/
  <dataset_name>/
    train/
      class_1/
      class_2/
      ...
    val/
      class_1/
      class_2/
      ...
```

For CUB conversion support, see convert_dataset.py.

## 4. Repository Structure

```text
INTR/
  main.py                        # original train/eval entrypoint
  engine.py                      # train/eval loops
  finetune_k_queries.py          # k-query finetuning (resume + quick eval options)
  demo_single_image.py           # single image inference + heatmap output
  convert_dataset.py             # CUB metadata -> ImageFolder conversion
  test_k_queries.py              # sanity checks for k-query behavior
  demo.ipynb                     # notebook demo
  models/
    intr.py                      # INTR model + K-query aggregation
    backbone.py
    transformer.py
    position_encoding.py
  datasets/
    build.py                     # ImageFolder + transforms
    transforms.py
    constants.py
  tools/
    visualization.py             # attention visualization script
  util/
    misc.py                      # metrics, class_accuracy, helpers
  output/                        # generated during experiments (gitignored)
  checkpoints/                   # model weights (gitignored)
```

## 5. Changes Implemented in This Project

### 5.1 K queries per class

Added in main.py, models/intr.py, and tools/visualization.py:
- --k_queries_per_class
- --query_aggregation {max, mean, sum}

Behavior:
- Total queries = num_classes * k_queries_per_class
- Query logits are reshaped to [batch, num_classes, k]
- Aggregated class score is used for loss and accuracy

Outputs in model forward now include:
- query_logits: class-level logits [batch, num_classes]
- all_query_logits: per-query logits [batch, num_queries]
- query_logits_per_class: grouped logits [batch, num_classes, k]

### 5.2 Visualization adapted for K-query models

tools/visualization.py now:
- Selects the most important query within each class group.
- Supports k-query indexing logic for heatmaps.
- Includes Windows path compatibility fixes.

### 5.3 Finetuning script with resume and quick-test options

finetune_k_queries.py includes:
- Initialization from pretrained k=1 checkpoint to k>1 query embeddings.
- Freeze-all except query_embed and presence_vector.
- Resume optimizer/scheduler/epoch from checkpoint.
- Skip redundant pre-finetuning evaluation on resume.
- CUDA cache clearing before training epochs.
- Additional quick-test controls:
  - --eval_batch_size
  - --max_eval_samples
  - --max_train_samples
  - --save_init_checkpoint

### 5.4 Single-image demo workflow

demo_single_image.py supports:
- Any input image size.
- INTR preprocessing pipeline.
- Prediction + top-k class output.
- Heatmap overlay generation from decoder attention.
- JSON output for reproducible inference results.

## 6. How to Run

### 6.1 Baseline evaluation

```bash
python main.py \
  --eval \
  --resume checkpoints/intr_checkpoint_cub_detr_r50.pth \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --batch_size 4
```

### 6.2 K-query finetuning from pretrained checkpoint

```bash
python finetune_k_queries.py \
  --pretrained checkpoints/intr_checkpoint_cub_detr_r50.pth \
  --k_queries_per_class 3 \
  --query_aggregation max \
  --epochs 4 \
  --batch_size 4 \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --output_sub_dir k3_finetune
```

### 6.3 Resume k-query finetuning

```bash
python finetune_k_queries.py \
  --resume output/CUB_200_2011_formatted/k3_finetune/checkpoint.pth \
  --k_queries_per_class 3 \
  --query_aggregation max \
  --epochs 4 \
  --batch_size 4 \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --output_sub_dir k3_finetune
```

### 6.4 Quick subset-only eval for k-query initialization

```bash
python finetune_k_queries.py \
  --pretrained checkpoints/intr_checkpoint_cub_detr_r50.pth \
  --k_queries_per_class 7 \
  --query_aggregation max \
  --eval \
  --save_init_checkpoint \
  --max_eval_samples 800 \
  --eval_batch_size 1 \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --output_sub_dir k7_finetune_minitest
```

### 6.5 Single-image demo with heatmap

```bash
python demo_single_image.py \
  --image_path demo_image/Yellow_warbler_(82905).jpg \
  --checkpoint output/CUB_200_2011_formatted/k3_finetune/checkpoint.pth \
  --dataset_name CUB_200_2011_formatted \
  --k_queries_per_class 3 \
  --query_aggregation max \
  --output_dir output/demo_single_k3
```

## 7. Experimental Notes from This Project

Observed metrics on CUB pipeline in this workspace:
- Baseline checkpoint evaluation: acc1 71.86, acc5 89.39
- K=3 finetune epoch 0 snapshot: acc1 about 71.70, acc5 about 89.51
- K=5 subset minitest (800 eval samples): acc1 71.25, acc5 88.62
- K=7 subset minitest (800 eval samples): acc1 70.50, acc5 88.62

Important practical note:
- On memory-constrained GPUs, evaluation can fail at large eval settings.
- Use --eval_batch_size 1 and --max_eval_samples for stable quick checks.

## 8. Novelty in This Extended Version

The main novelty added in this project is moving from one-query-per-class to multi-query-per-class classification with explicit query aggregation and interpretable query selection.

Why this matters:
- A single class can be represented by multiple learned query prototypes.
- Different queries can specialize to different visual parts, poses, or context.
- Aggregation provides a robust class score while retaining per-query interpretability.
- Visualization can reveal which specific query instance drove the final prediction.

In short, this extension keeps the original INTR interpretability principle but increases representational flexibility per class, making the model more expressive for fine-grained categories.

## 9. Related Documentation in This Repo

- K_QUERIES_PER_CLASS_CHANGES.md
- QUICK_START_K_QUERIES.md
- VISUALIZATION_GUIDE.md
- EVALUATION_RESULTS.md

## 10. Acknowledgment

INTR is inspired by DETR (DEtection TRansformer):
- https://github.com/facebookresearch/detr

## 11. Citation

```bibtex
@inproceedings{paul2024simple,
  title={A Simple Interpretable Transformer for Fine-Grained Image Classification and Analysis},
  author={Paul, Dipanjyoti and Chowdhury, Arpita and Xiong, Xinqi and Chang, Feng-Ju and Carlyn, David and Stevens, Samuel and Provost, Kaiya and Karpatne, Anuj and Carstens, Bryan and Rubenstein, Daniel and Stewart, Charles and Berger-Wolf, Tanya and Su, Yu and Chao, Wei-Lun},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

