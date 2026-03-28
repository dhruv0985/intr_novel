# INTR Model Evaluation Results

## Evaluation Details

**Date & Time**: February 26, 2026 at 01:13:37  
**Model**: INTR (Interpretable Transformer)  
**Checkpoint**: `checkpoints/intr_checkpoint_cub_detr_r50.pth`  
**Dataset**: CUB_200_2011 (Caltech-UCSD Birds-200-2011)  
**Dataset Path**: `datasets/CUB_200_2011_formatted`

## Dataset Statistics

- **Total Images**: 11,788
- **Number of Classes**: 200 bird species
- **Train Images**: 5,994
- **Validation Images**: 5,794

## Model Configuration

- **Backbone**: ResNet-50
- **Encoder Layers**: 6
- **Decoder Layers**: 6
- **Hidden Dimension**: 256
- **Number of Queries**: 200
- **Batch Size**: 4
- **Device**: CUDA (NVIDIA GeForce RTX 3050 Laptop GPU)

## Evaluation Results

| Metric | Value |
|--------|-------|
| **Loss** | 3.557 |
| **Top-1 Accuracy** | **71.86%** |
| **Top-5 Accuracy** | **89.39%** |

## Command Used

```bash
python main.py \
  --eval \
  --resume checkpoints/intr_checkpoint_cub_detr_r50.pth \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --batch_size 4
```

## Output Files

- **Log File**: `output/CUB_200_2011_formatted/output_sub/log.txt`
- **File Timestamp**: February 26, 2026 01:13:37
- **Python Version**: 3.8.6 (64-bit)
- **PyTorch Version**: 2.4.1+cu118
- **CUDA Version**: 12.7

## Environment Details

- **Virtual Environment**: `INTR/intr/` (Python 3.8.6)
- **GPU Memory Usage**: ~2GB (out of 4GB available)
- **Evaluation Time**: ~50 minutes
- **Number of Batches Processed**: 1,449

## Key Findings

The model achieved **71.86% top-1 accuracy** on the CUB-200-2011 validation set, which is a strong result for fine-grained bird species classification. The top-5 accuracy of **89.39%** indicates that the correct species is in the top 5 predictions almost 90% of the time.

---

*Generated on: February 26, 2026*
