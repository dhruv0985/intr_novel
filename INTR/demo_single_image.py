import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch

import datasets.transforms as T
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser("INTR single-image demo")

    parser.add_argument("--image_path", required=True, type=str,
                        help="Path to input image (any size)")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="output/demo_single", type=str,
                        help="Directory to save outputs")
    parser.add_argument("--classes_file", default="demo_image/classes.txt", type=str,
                        help="Class names file (optional)")

    parser.add_argument("--dataset_name", default="CUB_200_2011_formatted", type=str)
    parser.add_argument("--dataset_path", default="datasets", type=str)

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dec_layer_index", default=5, type=int,
                        help="Decoder layer index for heatmap (0-based)")
    parser.add_argument("--topk", default=5, type=int,
                        help="Top-k predictions to print/save")

    # Model params (must match training config)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--backbone", default="resnet50", type=str)
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned"))
    parser.add_argument("--enc_layers", default=6, type=int)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=2048, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--num_queries", default=200, type=int)
    parser.add_argument("--pre_norm", action="store_true")

    parser.add_argument("--k_queries_per_class", default=None, type=int,
                        help="If omitted, inferred from checkpoint")
    parser.add_argument("--query_aggregation", default="max", type=str, choices=["max", "mean", "sum"])

    # Not used directly, but required by model builder compatibility
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://", type=str)

    return parser


def make_val_transform():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])


def parse_classes_file(path):
    path_obj = Path(path)
    if not path_obj.exists():
        return None

    names = []
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                names.append(parts[0])
            else:
                names.append(parts[1])
    return names if names else None


def infer_k_from_checkpoint(checkpoint_obj, state_dict, num_classes):
    if isinstance(checkpoint_obj, dict) and "args" in checkpoint_obj:
        ckpt_args = checkpoint_obj["args"]
        if hasattr(ckpt_args, "k_queries_per_class"):
            return int(ckpt_args.k_queries_per_class)

    if "query_embed.weight" in state_dict:
        total_queries = int(state_dict["query_embed.weight"].shape[0])
        if num_classes > 0 and total_queries % num_classes == 0:
            return max(1, total_queries // num_classes)

    return 1


def to_target_format(label_idx, device):
    return [{"image_label": torch.tensor([label_idx], dtype=torch.int64, device=device)}]


def overlay_heatmap(avg_attention, encoder_output, image_bgr):
    h_feat = encoder_output.shape[2]
    w_feat = encoder_output.shape[3]
    heatmap = avg_attention.reshape(h_feat, w_feat).detach().cpu().numpy()

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    alpha = 0.5
    blended = (alpha * image_bgr + (1.0 - alpha) * heatmap_color).astype(np.uint8)
    return blended, heatmap_uint8


@torch.no_grad()
def run_demo(args):
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint_obj = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
        state_dict = checkpoint_obj["model"]
    else:
        state_dict = checkpoint_obj

    num_classes = 200 if (args.dataset_name == "cub" or "CUB" in args.dataset_name) else args.num_queries
    if args.k_queries_per_class is None:
        args.k_queries_per_class = infer_k_from_checkpoint(checkpoint_obj, state_dict, num_classes)

    model, _ = build_model(args)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.to(device)
    model.eval()

    classes = parse_classes_file(args.classes_file)

    pil_img = Image.open(image_path).convert("RGB")
    transform = make_val_transform()
    tensor_img, _ = transform(pil_img, None)
    tensor_img = tensor_img.to(device)

    outputs, encoder_output, _, _, avg_attention_scores = model([tensor_img])

    class_logits = outputs["query_logits"][0]
    probs = torch.softmax(class_logits, dim=0)
    topk = min(args.topk, probs.numel())
    top_probs, top_indices = torch.topk(probs, k=topk)

    pred_class = int(top_indices[0].item())
    pred_prob = float(top_probs[0].item())

    k = int(getattr(model, "k_queries_per_class", 1))
    all_query_logits = outputs["all_query_logits"][0]
    q_start = pred_class * k
    q_end = q_start + k
    best_query_offset = int(torch.argmax(all_query_logits[q_start:q_end]).item())
    best_query_index = q_start + best_query_offset

    attention_for_query = avg_attention_scores[args.dec_layer_index, 0, best_query_index, :]

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image for heatmap: {image_path}")

    heatmap_overlay, raw_heatmap = overlay_heatmap(attention_for_query, encoder_output, image_bgr)

    heatmap_path = output_dir / f"{image_path.stem}_heatmap.jpg"
    raw_heatmap_path = output_dir / f"{image_path.stem}_heatmap_raw.jpg"
    cv2.imwrite(str(heatmap_path), heatmap_overlay)
    cv2.imwrite(str(raw_heatmap_path), cv2.resize(raw_heatmap, (image_bgr.shape[1], image_bgr.shape[0])))

    topk_predictions = []
    for i in range(topk):
        cls_idx = int(top_indices[i].item())
        cls_prob = float(top_probs[i].item())
        cls_name = classes[cls_idx] if classes and cls_idx < len(classes) else f"class_{cls_idx}"
        topk_predictions.append({
            "class_index": cls_idx,
            "class_name": cls_name,
            "probability": cls_prob,
        })

    pred_name = classes[pred_class] if classes and pred_class < len(classes) else f"class_{pred_class}"
    result = {
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "k_queries_per_class": k,
        "query_aggregation": args.query_aggregation,
        "predicted_class_index": pred_class,
        "predicted_class_name": pred_name,
        "predicted_probability": pred_prob,
        "best_query_index_for_prediction": best_query_index,
        "topk": topk_predictions,
        "heatmap_overlay": str(heatmap_path),
        "heatmap_raw": str(raw_heatmap_path),
    }

    result_path = output_dir / f"{image_path.stem}_prediction.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nPrediction complete")
    print(f"Image: {image_path}")
    print(f"Predicted class: {pred_name} ({pred_class})")
    print(f"Probability: {pred_prob:.4f}")
    print(f"Heatmap overlay saved to: {heatmap_path}")
    print(f"Results JSON saved to: {result_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    cli_args = parser.parse_args()
    run_demo(cli_args)
