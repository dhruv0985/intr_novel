# Quick Start Guide: Using K Queries Per Class

## ✅ Implementation Complete!

Your INTR model now supports **K queries per class** instead of just 1 query per class. The test script confirms all components work correctly.

## 🎯 What Changed?

### Architecture
- **Before:** 200 queries (1 per class) for CUB dataset
- **Now:** 200 × K queries (K per class) for CUB dataset
- **Example:** With K=3, you get 600 total queries

### Prediction
Each class has K queries. The final class score is the **aggregation** (max/mean/sum) of those K query logits.

### Visualization  
The system **automatically selects** the most important query (highest logit) among the K queries for each class.

---

## 📋 Usage Commands

### 1. **Training** from Scratch with K=3

```bash
python main.py \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --k_queries_per_class 3 \
  --query_aggregation max \
  --batch_size 8 \
  --epochs 140 \
  --output_dir output \
  --output_sub_dir k3_max
```

**Parameters:**
- `--k_queries_per_class 3` → Use 3 queries per class (600 total)
- `--query_aggregation max` → Use max pooling across K queries
- Other options: `mean`, `sum`

### 2. **Evaluation** with Existing Checkpoint (K=1, backward compatible)

```bash
python main.py \
  --eval \
  --resume checkpoints/intr_checkpoint_cub_detr_r50.pth \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --k_queries_per_class 1 \
  --batch_size 4
```

**Note:** Old checkpoints use K=1, so always specify `--k_queries_per_class 1` when loading them!

### 3. **Visualization** with K=3 (shows most important query)

```bash
python -m tools.visualization \
  --eval \
  --resume checkpoints/intr_k3_checkpoint.pth \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --class_index 0 \
  --k_queries_per_class 3 \
  --query_aggregation max \
  --gt_query_heads 1 \
  --sim_query_heads 1 \
  --output_dir output/visualization_k3
```

**What you'll see:**
- For class 0 (with queries 0, 1, 2), if query 1 has the highest logit, the visualization will show query 1's attention
- Automatically picks the most discriminative query!

---

## 🧪 Testing Your Setup

Run the test script to verify everything works:

```bash
python test_k_queries.py
```

**Expected output:**
```
============================================================
✓ All tests passed! Implementation is correct.
============================================================
```

---

## 🔧 Hyperparameter Settings

### Number of Queries (K)

| K Value | Total Queries (CUB) | Use Case |
|---------|---------------------|----------|
| 1 | 200 | Original INTR (baseline) |
| 2 | 400 | Simple ensemble |
| 3 | 600 | Good balance ✅ |
| 5 | 1000 | More diversity |
| 10 | 2000 | Maximum diversity (high memory) |

**Recommendation:** Start with K=3

### Aggregation Strategy

| Strategy | Formula | Best For |
|----------|---------|----------|
| `max` | max(q₁, q₂, ..., qₖ) | Confidence-based ✅ |
| `mean` | (q₁ + q₂ + ... + qₖ) / K | Ensemble average |
| `sum` | q₁ + q₂ + ... + qₖ | Total evidence |

**Recommendation:** Start with `max`

---

## 📊 Expected Behavior

### Query Organization (K=3 example)

```
Class 0 (Black-footed Albatross):
  - Query 0: Learns one aspect (e.g., bird in flight)
  - Query 1: Learns another aspect (e.g., bird on ground)
  - Query 2: Learns third aspect (e.g., close-up)

Class 1 (Laysan Albatross):
  - Query 3: First aspect
  - Query 4: Second aspect
  - Query 5: Third aspect

...and so on for all 200 classes
```

### Prediction Flow (K=3, max aggregation)

```
For an input image of Class 0:
1. Model outputs 600 query logits
2. Queries 0-2 (Class 0) produce: [2.1, 3.5, 1.8]
3. Max aggregation: score for Class 0 = max(2.1, 3.5, 1.8) = 3.5
4. Compare with other 199 classes' max scores
5. Predict the class with highest max score
```

### Visualization Flow (K=3)

```
For visualizing Class 0:
1. Get logits for queries 0-2: [2.1, 3.5, 1.8]
2. Query 1 has highest logit (3.5)
3. Visualize attention maps of Query 1
4. This shows which visual aspect was most important!
```

---

## ⚠️ Important Notes

### 1. Checkpoint Compatibility

**Problem:** Old checkpoints (K=1) cannot be directly loaded for K>1

**Solutions:**
- ✅ **Option A:** Train from scratch with desired K
- ✅ **Option B:** Use K=1 with old checkpoints: `--k_queries_per_class 1`
- ⚠️ **Option C:** Implement custom query initialization (requires code modification)

### 2. Memory Usage

With K=3, you have 3× more queries:
- More memory needed
- Reduce batch size if needed
- Example: `--batch_size 4` instead of 12

### 3. Training Time

K>1 may need more epochs to converge:
- Each query needs to specialize
- Consider increasing epochs or adjusting learning rate

---

## 🎨 Example Experiment

### Experiment: Compare K=1 vs K=3

**Step 1:** Evaluate baseline (K=1)
```bash
python main.py --eval --resume checkpoints/intr_checkpoint_cub_detr_r50.pth \
  --dataset_path datasets --dataset_name CUB_200_2011_formatted \
  --k_queries_per_class 1 --batch_size 4
```

**Expected:** ~71.86% accuracy

**Step 2:** Train with K=3
```bash
python main.py \
  --dataset_path datasets --dataset_name CUB_200_2011_formatted \
  --k_queries_per_class 3 --query_aggregation max \
  --batch_size 8 --epochs 140
```

**Step 3:** Evaluate K=3
```bash
python main.py --eval --resume output/CUB_200_2011_formatted/k3_max/checkpoint.pth \
  --dataset_path datasets --dataset_name CUB_200_2011_formatted \
  --k_queries_per_class 3 --query_aggregation max --batch_size 4
```

**Step 4:** Visualize both
```bash
# K=1 visualization
python -m tools.visualization --eval \
  --resume checkpoints/intr_checkpoint_cub_detr_r50.pth \
  --dataset_path datasets --dataset_name CUB_200_2011_formatted \
  --class_index 0 --k_queries_per_class 1 \
  --gt_query_heads 1 --output_dir output/viz_k1

# K=3 visualization  
python -m tools.visualization --eval \
  --resume output/CUB_200_2011_formatted/k3_max/checkpoint.pth \
  --dataset_path datasets --dataset_name CUB_200_2011_formatted \
  --class_index 0 --k_queries_per_class 3 --query_aggregation max \
  --gt_query_heads 1 --output_dir output/viz_k3
```

**Compare:** Does K=3 improve accuracy? Which query gets selected for visualization?

---

## 📚 Files Modified

1. ✅ [main.py](main.py) - Added hyperparameters
2. ✅ [models/intr.py](models/intr.py) - Modified architecture
3. ✅ [tools/visualization.py](tools/visualization.py) - Auto-select important query

## 📖 Documentation

- 📄 [K_QUERIES_PER_CLASS_CHANGES.md](K_QUERIES_PER_CLASS_CHANGES.md) - Detailed technical explanation
- 🧪 [test_k_queries.py](test_k_queries.py) - Test script
- 📋 This file - Quick start guide

---

## ❓ Troubleshooting

### Error: "Size mismatch when loading checkpoint"
**Cause:** Checkpoint has different number of queries  
**Fix:** Ensure `--k_queries_per_class` matches the checkpoint's K value

### Error: "Out of memory"
**Cause:** Too many queries or large batch size  
**Fix:** Reduce `--batch_size` or use smaller K

### Accuracy is lower with K>1
**Cause:** Queries might need more training to specialize  
**Fix:** Increase epochs, adjust learning rate, or try different aggregation

---

## 🎯 Quick Commands Reference

```bash
# Test implementation
python test_k_queries.py

# Train with K=3
python main.py --k_queries_per_class 3 --query_aggregation max --batch_size 8

# Eval with K=1 (old checkpoint)
python main.py --eval --resume old.pth --k_queries_per_class 1

# Eval with K=3 (new checkpoint)
python main.py --eval --resume new.pth --k_queries_per_class 3 --query_aggregation max

# Visualize with K=3
python -m tools.visualization --eval --resume new.pth --k_queries_per_class 3 --class_index 0
```

---

**Ready to use!** Start with K=3 and max aggregation for best results. 🚀
