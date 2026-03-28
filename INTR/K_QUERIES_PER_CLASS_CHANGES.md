# K Queries Per Class - Architecture Modification

## Overview

Modified the INTR architecture to support **K queries per class** instead of 1 query per class. This allows the model to learn multiple diverse representations for each class, potentially capturing different visual aspects or variations within the same class.

## Changes Made

### 1. **New Hyperparameters** (main.py, visualization.py)

Added three new command-line arguments:

```python
--k_queries_per_class <int>      # Number of queries per class (default: 1)
--query_aggregation <str>        # How to aggregate K queries: 'max', 'mean', or 'sum' (default: 'max')
```

**Examples:**
- `--k_queries_per_class 1` → Original INTR (1 query per class, 200 total queries for CUB)
- `--k_queries_per_class 3` → 3 queries per class, 600 total queries for CUB
- `--k_queries_per_class 5` → 5 queries per class, 1000 total queries for CUB

### 2. **Model Architecture Changes** (models/intr.py)

#### Modified INTR Class:

**Before:**
```python
def __init__(self, args, backbone, transformer, num_queries):
    self.num_queries = num_queries  # e.g., 200 for CUB
    self.query_embed = nn.Embedding(num_queries, hidden_dim)
```

**After:**
```python
def __init__(self, args, backbone, transformer, num_classes, k_queries_per_class=1):
    self.num_classes = num_classes  # e.g., 200 for CUB
    self.k_queries_per_class = k_queries_per_class  # e.g., 3
    self.num_queries = num_classes * k_queries_per_class  # e.g., 600 total queries
    self.query_aggregation = args.query_aggregation  # 'max', 'mean', or 'sum'
    self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
```

**Key points:**
- Total queries = `num_classes × k_queries_per_class`
- Queries 0-2 → Class 0, Queries 3-5 → Class 1, etc. (if k=3)
- Each group of K queries learns different aspects of the same class

#### Modified Forward Pass:

**Before:**
```python
query_logits = self.presence_vector(hs[-1]).squeeze(dim=-1)
out = {'query_logits': query_logits}  # [batch_size, 200]
```

**After:**
```python
query_logits = self.presence_vector(hs[-1]).squeeze(dim=-1)  # [batch_size, num_queries]

# Reshape to [batch_size, num_classes, k_queries_per_class]
query_logits_reshaped = query_logits.view(batch_size, self.num_classes, self.k_queries_per_class)

# Aggregate across K queries
if self.query_aggregation == 'max':
    class_logits, _ = torch.max(query_logits_reshaped, dim=2)
elif self.query_aggregation == 'mean':
    class_logits = torch.mean(query_logits_reshaped, dim=2)
elif self.query_aggregation == 'sum':
    class_logits = torch.sum(query_logits_reshaped, dim=2)

out = {
    'query_logits': class_logits,  # [batch_size, num_classes] - for loss/accuracy
    'all_query_logits': query_logits,  # [batch_size, num_queries] - for visualization
    'query_logits_per_class': query_logits_reshaped  # [batch_size, num_classes, k]
}
```

**Aggregation strategies:**
- **max**: Use the highest scoring query per class (most confident representation)
- **mean**: Average all K queries (ensemble)
- **sum**: Sum all K queries (total evidence)

### 3. **Loss & Training** (No changes needed!)

The loss computation in `SetCriterion` and training in `engine.py` work unchanged because:
- `outputs['query_logits']` still has shape `[batch_size, num_classes]`
- Cross-entropy loss expects this exact format
- The aggregation is transparent to the loss function

### 4. **Visualization Changes** (tools/visualization.py)

Modified to **automatically select the most important query** for visualization:

#### In `evaluate()` function:

```python
# For ground truth class, find which of the K queries had highest logit
gt_start_idx = image_label * k_queries_per_class
gt_end_idx = gt_start_idx + k_queries_per_class
gt_class_query_logits = all_query_logits[gt_start_idx:gt_end_idx]
gt_best_query_offset = torch.argmax(gt_class_query_logits).item()
gt_query_index = gt_start_idx + gt_best_query_offset

# For similar classes, also find best query
for class_idx in similar_classes:
    start_idx = class_idx * k_queries_per_class
    end_idx = start_idx + k_queries_per_class
    class_query_logits = all_query_logits[start_idx:end_idx]
    best_query_offset = torch.argmax(class_query_logits).item()
    best_query_idx = start_idx + best_query_offset
    similar_query_indices.append(best_query_idx)
```

**What this means:**
- Instead of visualizing a fixed query index, we now visualize the **most confident query**
- Each class's visualization shows the query that contributed most to the prediction
- This reveals which visual aspect the model found most discriminative

## Usage Examples

### Training with K=3 queries per class:

```bash
python main.py \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --k_queries_per_class 3 \
  --query_aggregation max \
  --batch_size 8 \
  --epochs 140
```

### Evaluation with K queries:

```bash
python main.py \
  --eval \
  --resume checkpoints/intr_checkpoint_k3.pth \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --k_queries_per_class 3 \
  --query_aggregation max \
  --batch_size 4
```

### Visualization with K queries:

```bash
python -m tools.visualization \
  --eval \
  --resume checkpoints/intr_checkpoint_k3.pth \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --class_index 0 \
  --k_queries_per_class 3 \
  --query_aggregation max \
  --gt_query_heads 1 \
  --sim_query_heads 1 \
  --output_dir output/visualization_k3
```

**Note:** The visualization will automatically show the most important query among the 3 queries for each class!

## Understanding the Output

### With K=3 queries per class:

**Query organization:**
```
Queries 0-2   → Class 0 (Black-footed Albatross)
Queries 3-5   → Class 1 (Laysan Albatross)
Queries 6-8   → Class 2 (Sooty Albatross)
...
Queries 597-599 → Class 199
```

**Example prediction flow:**
1. Model produces 600 query logits (200 classes × 3 queries)
2. For Class 0: logits = [2.1, 3.5, 1.8]
3. With `--query_aggregation max`: Class 0 score = max(2.1, 3.5, 1.8) = 3.5
4. For visualization: Query 1 (index=1) is selected as it had the highest logit (3.5)

## Benefits of K Queries Per Class

1. **Diversity**: Each query can learn different visual aspects:
   - Query 0 → Bird in flight
   - Query 1 → Bird perched
   - Query 2 → Close-up of head
   
2. **Robustness**: Multiple queries provide ensemble-like behavior

3. **Interpretability**: Visualization shows which aspect was most important for prediction

4. **Flexibility**: Can compare different K values and aggregation strategies

## Backward Compatibility

Setting `--k_queries_per_class 1` (default) gives **identical behavior** to the original INTR:
- 1 query per class
- No aggregation needed
- Same performance and visualizations

## Testing the Implementation

### Quick test with K=2:

```bash
# Evaluate with K=2 queries per class
python main.py \
  --eval \
  --resume checkpoints/intr_checkpoint_cub_detr_r50.pth \
  --dataset_path datasets \
  --dataset_name CUB_200_2011_formatted \
  --k_queries_per_class 2 \
  --query_aggregation max \
  --batch_size 4
```

**Expected behavior:**
- Model will use 400 total queries (200 classes × 2)
- For each class, the max of 2 query logits will be used
- Accuracy should be computed correctly
- Visualization will show the more confident query

## Files Modified

1. ✅ `main.py` - Added hyperparameters
2. ✅ `models/intr.py` - Modified architecture and forward pass
3. ✅ `tools/visualization.py` - Updated to use most important query
4. ⚠️ `engine.py` - No changes needed (transparent)
5. ⚠️ Loss computation - No changes needed (transparent)

## Potential Issues & Solutions

### Issue 1: Loading old checkpoints
**Problem:** Old checkpoints have `num_queries=200`, new model expects `num_queries=num_classes*k`

**Solution:** When loading old checkpoint with k>1, the model will have mismatched query embeddings. You need to either:
1. Train from scratch with new K value
2. Or initialize new queries by copying/interpolating old ones (requires custom loading code)

### Issue 2: Memory usage
**Problem:** K=5 means 5× more queries (1000 queries for CUB), more memory

**Solution:** 
- Reduce batch size
- Use gradient checkpointing
- Or use smaller K values (2-3 usually sufficient)

### Issue 3: Ensuring compatibility
**Problem:** Want to test with old checkpoints (k=1)

**Solution:** Always specify `--k_queries_per_class 1` when loading old checkpoints:
```bash
python main.py --eval --resume old_ckpt.pth --k_queries_per_class 1 ...
```

## Next Steps / Experiments

1. **Train with different K values** (2, 3, 5) and compare accuracy
2. **Try different aggregation strategies** (max vs mean vs sum)
3. **Visualize all K queries** per class to see what different aspects they learn
4. **Analyze query specialization** - do queries specialize on pose, background, etc?
5. **Use query diversity as a regularization term** in the loss

---

**Implementation Date:** February 26, 2026  
**Author:** Modified INTR architecture for multi-query learning
