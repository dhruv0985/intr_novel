# INTR Model Visualization Guide

## Generated: February 26, 2026

## Overview
This document explains the attention map visualizations generated for the INTR model on the CUB-200 dataset.

## What Was Generated

### Visualizations for Class 0 (Black-footed Albatross)
- **Location**: `output/visualization/visualization/`
- **Number of Images Visualized**: 30 validation images
- **Visualization Types**: 
  - Ground truth query attention heads
  - Top-5 similar query attention maps

## Directory Structure

Each image has its own folder with the following structure:
```
Black_Footed_Albatross_XXXX_YYYYYY/
├── heads/                          # Ground Truth Query Visualizations
│   ├── Black_Footed_Albatross_XXXX_YYYYYY.png  # Original image
│   ├── result_head_0.png          # Attention heatmap for head 0
│   ├── result_head_1.png          # Attention heatmap for head 1
│   ├── ...                        # Heads 2-6
│   ├── result_head_7.png          # Attention heatmap for head 7
│   ├── avg_head.png               # Average of all 8 heads
│   └── concatenated_False.png     # All heads side-by-side (False = incorrect prediction)
│
├── head_avg/                       # Similar Queries Average Attention
│   ├── avg_head_0.png             # Query 0 (correct class)
│   ├── avg_head_1.png             # Most similar query
│   ├── avg_head_2.png             # 2nd most similar
│   ├── avg_head_44.png            # 3rd most similar (class 44)
│   └── avg_head_71.png            # 4th most similar (class 71)
│
├── head_0/                         # Individual attention heads for similar queries
│   ├── Black_Footed_Albatross_XXXX_YYYYYY.png
│   ├── result_query_0.png
│   ├── ...
│   └── concatenated_False.png
│
└── head_1/ ... head_7/            # Same structure for each head
```

## Understanding the Visualizations

### 1. **Ground Truth Query Heads** (`heads/` folder)
- Shows what the model's attention heads focus on for the CORRECT class
- Each of the 8 attention heads learns to focus on different features
- **Red/Yellow** regions = High attention (what the model is looking at)
- **Blue/Purple** regions = Low attention (what the model ignores)

### 2. **Similar Queries** (`head_avg/` folder)
- Shows the top 5 queries that produced the highest confidence scores
- Query 0 is always the correct class (Black-footed Albatross)
- Other queries show which classes the model thinks are similar
- Helps understand attribute similarities between classes

### 3. **Individual Head Analysis** (`head_0/` through `head_7/`)
- Detailed attention maps for each attention head
- Shows how each head focuses on different parts for similar queries

## How to Use These Visualizations

### To view all images for one bird:
```powershell
cd output\visualization\visualization\Black_Footed_Albatross_0001_796111\heads
# Open the images in your image viewer
```

### To compare attention across different birds:
Look at the `concatenated_*.png` files which show all 8 heads side-by-side.

### To understand model confusion:
Check the `head_avg/` folder to see which other classes (queries) the model finds similar.

## Running Visualizations for Other Classes

To visualize a different bird class:
```bash
python -m tools.visualization \
  --eval \
  --resume checkpoints/intr_checkpoint_cub_detr_r50.pth \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --class_index <CLASS_NUMBER> \
  --gt_query_heads 1 \
  --sim_query_heads 1 \
  --output_dir output/visualization_class_<CLASS_NUMBER>
```

### Class Index Reference:
- 0 = Black-footed Albatross
- 1 = Laysan Albatross
- 2 = Sooty Albatross
- 16 = Cardinal
- ... (see `datasets/CUB_200_2011/classes.txt` for full list, subtract 1 for index)

## Visualization Parameters

- `--class_index`: Which class to visualize (0-199)
- `--dec_layer_index`: Which decoder layer to visualize (default: 5, last layer)
- `--top_q`: Number of top similar queries (default: 5)
- `--gt_query_heads`: Visualize ground truth query heads (1 = yes, 0 = no)
- `--sim_query_heads`: Visualize similar queries (1 = yes, 0 = no)

## Example Findings

Based on the generated visualizations for Black-footed Albatross:
1. The model attends to bird body, head, and wing features
2. Different attention heads focus on different parts:
   - Some heads focus on the head/beak
   - Some focus on body/wings
   - Some focus on texture/background
3. Similar queries (other classes) likely share visual attributes

## Next Steps

1. **View the visualizations**: Open the PNG files in `output/visualization/visualization/`
2. **Compare correct vs incorrect predictions**: Look at images with `concatenated_True.png` vs `concatenated_False.png`
3. **Analyze confusion patterns**: Check which classes appear as similar queries
4. **Visualize more classes**: Run the visualization script for other interesting classes

## Proving You Ran This

To prove you generated these visualizations, you can:
1. Take screenshots of the visualization folders and images
2. Show the timestamp on the folders (created today)
3. Share sample heatmap images
4. Include this guide with the timestamps
5. Create a video walkthrough of the visualization folders
