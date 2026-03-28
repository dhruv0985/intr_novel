# ------------------------------------------------------------------------
# INTR
# Copyright (c) 2023 Imageomics Paul. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import cv2 
import json
import time
import math
import shutil
import random
import argparse
import datetime

import numpy as np
from PIL import Image
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch
from scipy.ndimage import gaussian_filter
import datasets.transforms as T

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1.00e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int, choices=[1])

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # visualization parameters
    parser.add_argument('--class_index', default=0, type=int,
                        help="a class number to visuaization")
    parser.add_argument('--dec_layer_index', default=5, type=int,
                        help="a layer number to visuaization")
    parser.add_argument('--top_q', default=5, type=int,
                        help="top `top_q' similar queries")
    parser.add_argument('--gt_query_heads', default=1, type=int,
                        help="print ground truth query heads for visualization")
    parser.add_argument('--sim_query_heads', default=1, type=int,
                        help="print similar queries heads for visualization")
    parser.add_argument('--test', default="val", type=str, choices=["val", "test"])
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer") #default=0.1
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--k_queries_per_class', default=1, type=int,
                        help='Number of queries per class (default: 1)')
    parser.add_argument('--query_aggregation', default='max', type=str, choices=['max', 'mean', 'sum'],
                        help='How to aggregate K queries per class for prediction')
    parser.add_argument('--pre_norm', action='store_true')

    # * Dataset parameters
    parser.add_argument('--dataset_name', default='cub') 
    parser.add_argument('--dataset_path', default='/path/to/datasets', type=str) 
    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')

    # * Device parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--eval', action='store_true')
    
    # * Distributed training parameters
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def get_image_filename(args, filename):
    image_path=os.path.join(args.dataset_path, args.dataset_name + '/' + args.test)
    image_filename = os.path.join(image_path, filename)
    return image_filename

def combine_images(path, pred_class):
    images = [os.path.join(path, image) for image in os.listdir(path) if image.endswith('.png')]
    imgs = [Image.open(image) for image in images]
    widths, heights = zip(*(img.size for img in imgs))

    total_width = sum(widths)
    max_height = max(heights)
    merged_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in imgs:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width
    merged_image.save(path + "/" + "concatenated_"+str(bool(pred_class))+".png")

def SuperImposeHeatmap(attention, input_image):
    alpha=0.5
    avg_heatmap_resized = cv2.resize(attention, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    avg_normalized_heatmap = (avg_heatmap_resized - np.min(avg_heatmap_resized)) / (np.max(avg_heatmap_resized) - np.min(avg_heatmap_resized))
    heatmap = (avg_normalized_heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.medianBlur(heatmap,15)
    heatmap =  cv2.GaussianBlur(heatmap, (15, 15), 0)
    result = (input_image *alpha  + heatmap * (1-alpha)).astype(np.uint8)
    return result

def visualize_heads(encoder_output, attention_score_gt_query, avg_attention_score_gt_query, prediction, img_id, image_file, des_dir):

    if not os.path.exists(os.path.join(des_dir, f"heads")):
        os.mkdir(os.path.join(des_dir, f"heads"))

    input_image = cv2.imread(image_file)
    input_image = cv2.resize(input_image, (0,0), fx=0.8, fy=0.8) 

    des_image_file=os.path.join(des_dir+ '/' + f"heads" + "/"  + str(img_id) + '.png')
    cv2.imwrite(des_image_file, input_image)

    for head_index in range(attention_score_gt_query.shape[1]):

        heatmap_head=attention_score_gt_query[:, head_index, :].reshape(encoder_output.shape[2], encoder_output.shape[3]).detach().cpu().numpy()
        result=SuperImposeHeatmap(heatmap_head, input_image)

        filename = des_dir + "/" + f"heads" + "/" + f"result_head_{head_index}.png"
        cv2.imwrite(filename, result)

        # To visualize avg. attention head
        if head_index==7: 
            avg_heatmap_head=avg_attention_score_gt_query.reshape(encoder_output.shape[2], encoder_output.shape[3]).detach().cpu().numpy()
            result=SuperImposeHeatmap(avg_heatmap_head, input_image)
            filename = des_dir + "/" + f"heads" + "/" + f"avg_head.png" 
            cv2.imwrite(filename,result)

    combine_images(des_dir + "/" + f"heads", prediction)


def visualize_queries(encoder_output, attention_score, avg_attention_score, prediction, img_id, image_file, des_dir, similar_queries): 

    if not os.path.exists(os.path.join(des_dir, f"head_avg")):
        os.mkdir(os.path.join(des_dir, f"head_avg"))
    input_image = cv2.imread(image_file)
    input_image = cv2.resize(input_image, (0,0), fx=0.8, fy=0.8)

    for query_index in (similar_queries):
        avg_attention_score_query=avg_attention_score[ :, query_index, :]
        avg_heatmap=avg_attention_score_query.reshape(encoder_output.shape[2], encoder_output.shape[3]).detach().cpu().numpy()
        result=SuperImposeHeatmap(avg_heatmap, input_image)

        filename = des_dir + "/" + f"head_avg" + "/" + f"avg_head_{query_index}.png" 
        cv2.imwrite(filename, result)

    for head_index in range (attention_score.shape[1]):
        if not os.path.exists(os.path.join(des_dir, f"head_{head_index}")):
            os.mkdir(os.path.join(des_dir, f"head_{head_index}"))

        des_image_file=os.path.join(des_dir+ '/'  + f"head_{head_index}" + "/"  + str(img_id) + '.png')
        cv2.imwrite(des_image_file, input_image) 

        attention_score_head=attention_score[:, head_index,: , :]

        for query_index in (similar_queries):
            heatmap_query=attention_score_head[:, query_index, :].reshape(encoder_output.shape[2], encoder_output.shape[3]).detach().cpu().numpy()
            result=SuperImposeHeatmap(heatmap_query, input_image)

            filename = des_dir + "/" + f"head_{head_index}" + "/" + f"result_query_{query_index}.png"
            cv2.imwrite(filename, result)

        combine_images(des_dir + "/" + f"head_{head_index}", prediction)


def visualization(args, filename, encoder_output, attention_scores, avg_attention_scores, similar_query_indices, gt_query_index, prediction):
    """
    This visualization function is for visualizing attention score for all the heads, similar queries, avg attention score etc.
        -- if gt_query_heads is True (i.e., 1), it will visualize attention score to all the heads of an image correspond to the ground truth (gt) query.
        -- if sim_query_heads is True (i.e., 1), it will visualize attention score to all the heads of an image correspond to `top_q' similar queries.
        -- gt_query_index: The actual query index (not class index) that had the highest logit for the ground truth class
        -- similar_query_indices: List of query indices (not class indices) for similar classes
    """
    output_dir = args.output_dir
    vis_dir = "visualization"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    vis_path = os.path.join(output_dir, vis_dir)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)

    if args.gt_query_heads==1:

        print ("visualizing ground truth query heads attention weights ...")

        attention_score=attention_scores[args.dec_layer_index, :, :, :, :]
        avg_attention_score=avg_attention_scores[args.dec_layer_index, :, :, :]

        query_index = gt_query_index  # Use the most important query for ground truth class
        img_id = filename.split("/")[-1][:-4]

        attention_score_gt_query=attention_score[:, :, query_index, :]
        avg_attention_score_gt_query=avg_attention_score[ :, query_index, :]

        if not os.path.exists(os.path.join(vis_path, str(img_id))):
            os.mkdir(os.path.join(vis_path, str(img_id)))
        des_dir=os.path.join(output_dir, vis_dir, str(img_id))

        image_file=get_image_filename(args, filename)
        visualize_heads(encoder_output, attention_score_gt_query, avg_attention_score_gt_query, prediction, img_id, image_file, des_dir)

    if args.sim_query_heads==1: 

        print ("visualizing similar queries heads attention weights ...")

        attention_score=attention_scores[args.dec_layer_index, :, :, :, :]
        avg_attention_score=avg_attention_scores[args.dec_layer_index, :, :, :]

        img_id = filename.split("/")[-1][:-4]
        
        if not os.path.exists(os.path.join(vis_path, str(img_id))):
            os.mkdir(os.path.join(vis_path, str(img_id)))
        des_dir=os.path.join(output_dir, vis_dir, str(img_id))

        image_file=get_image_filename(args, filename)
        visualize_queries(encoder_output, attention_score, avg_attention_score, prediction, img_id, image_file, des_dir, similar_query_indices)
            
            
@torch.no_grad()
def evaluate(args, model, data_loader, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Get k_queries_per_class from model or args
    k_queries_per_class = getattr(model, 'k_queries_per_class', 1)
    num_classes = getattr(model, 'num_classes', model.num_queries)

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)

        ## get the image file name
        filenames=[t["file_name"] for t in targets]
        # Handle both Windows and Unix paths
        full_path = filenames[0][0].replace("\\", "/")
        parts = full_path.split("/")
        filename = "/".join(parts[-2:])

        for t in targets:
            del t["file_name"]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        image_label = [item['image_label'].item() for item in targets][0]

        ## Visualize interpretation only for the given class_index
        if image_label==args.class_index: 

            outputs, encoder_output, _, attention_scores, avg_attention_scores = model(samples)
            
            # Get class-level logits for finding similar classes
            class_logits = outputs['query_logits'].flatten()  # [num_classes]
            
            # Get all query logits to find the most important query per class
            all_query_logits = outputs['all_query_logits'].flatten()  # [num_queries]
            
            # Find the top `args.top_q' similar classes
            _, similar_classes = torch.topk(class_logits, k=args.top_q)
            similar_classes = similar_classes.tolist()

            # In case of incorrect prediction, manually add the correct class
            if args.class_index not in similar_classes:
                similar_classes[-1] = args.class_index
            
            # For each similar class, find the most important query (highest logit)
            similar_query_indices = []
            for class_idx in similar_classes:
                # Get the K queries for this class
                start_idx = class_idx * k_queries_per_class
                end_idx = start_idx + k_queries_per_class
                class_query_logits = all_query_logits[start_idx:end_idx]
                
                # Find which query had the highest logit
                best_query_offset = torch.argmax(class_query_logits).item()
                best_query_idx = start_idx + best_query_offset
                similar_query_indices.append(best_query_idx)
            
            # Find the most important query for the ground truth class
            gt_start_idx = image_label * k_queries_per_class
            gt_end_idx = gt_start_idx + k_queries_per_class
            gt_class_query_logits = all_query_logits[gt_start_idx:gt_end_idx]
            gt_best_query_offset = torch.argmax(gt_class_query_logits).item()
            gt_query_index = gt_start_idx + gt_best_query_offset

            _ , _, prediction = utils.class_accuracy(outputs, targets, topk=(1, 1))
            visualization(args, filename, encoder_output, attention_scores, avg_attention_scores, 
                         similar_query_indices, gt_query_index, prediction) 


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    model, _ = build_model(args)
    model.to(device)

    dataset_val = build_dataset(image_set=args.test, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    if args.eval:
        evaluate(args, model, data_loader_val, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('INTR interpretation visualization script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
