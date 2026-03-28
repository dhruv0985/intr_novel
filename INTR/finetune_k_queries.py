# ------------------------------------------------------------------------
# INTR - Finetune K Queries Per Class
# Loads pretrained INTR checkpoint (k=1), initializes K queries per class
# by replicating old queries + noise, freezes backbone/transformer,
# and trains only query_embed + presence_vector (classifier).
# ------------------------------------------------------------------------

import os
import json
import time
import random
import datetime
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path

import datasets
import util.misc as utils
from models import build_model
from datasets import build_dataset
from engine import evaluate, train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('INTR K-query finetuning', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate for query_embed + classifier')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--eval_batch_size', default=0, type=int,
                        help='Validation batch size (0 means use --batch_size)')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Few epochs to finetune queries + classifier')
    parser.add_argument('--lr_drop', default=6, type=int)
    parser.add_argument('--lr_scheduler', default="CosineAnnealingLR", type=str,
                        choices=["StepLR", "CosineAnnealingLR"])
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=200, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # K queries
    parser.add_argument('--k_queries_per_class', default=3, type=int,
                        help='Number of queries per class')
    parser.add_argument('--query_aggregation', default='max', type=str,
                        choices=['max', 'mean', 'sum'])
    parser.add_argument('--noise_frac', default=0.1, type=float,
                        help='Noise fraction added to replicated queries')

    # Dataset
    parser.add_argument('--dataset_name', default='CUB_200_2011_formatted', type=str)
    parser.add_argument('--dataset_path', default='datasets', type=str)
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--output_sub_dir', default='k_query_finetune')
    parser.add_argument('--test', default='val', type=str, choices=['val', 'test'])
    parser.add_argument('--max_train_samples', default=0, type=int,
                        help='Use only first N train samples for quick experiments (0 = all)')
    parser.add_argument('--max_eval_samples', default=0, type=int,
                        help='Use only first N eval samples for quick experiments (0 = all)')

    # Pretrained checkpoint
    parser.add_argument('--pretrained', default='', type=str,
                        help='Path to pretrained INTR checkpoint (k=1)')
    parser.add_argument('--resume', default='', type=str,
                        help='Resume finetuning from a saved K-query checkpoint')

    # Device
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_init_checkpoint', action='store_true',
                        help='When used with --eval, save initialized model checkpoint before exit')

    # Distributed
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')

    return parser


def load_pretrained_and_init_k_queries(model, pretrained_path, k_queries_per_class, noise_frac=0.1):
    """
    Load pretrained checkpoint (k=1) and initialize K queries per class.
    
    Strategy:
    - All weights except query_embed are loaded directly from checkpoint
    - For query_embed: each original query [i] is replicated K times with small noise
      to create queries [i*K], [i*K+1], ..., [i*K+K-1]
    - presence_vector (classifier) weights are loaded as-is (same shape)
    """
    print(f"\nLoading pretrained checkpoint: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    pretrained_state = checkpoint['model']
    
    # Get pretrained query embeddings [num_classes, hidden_dim]
    old_query_weight = pretrained_state['query_embed.weight']
    num_classes, hidden_dim = old_query_weight.shape
    print(f"  Pretrained queries: {old_query_weight.shape} ({num_classes} classes, {hidden_dim}d)")
    
    # Create new query embeddings [num_classes * k, hidden_dim]
    new_num_queries = num_classes * k_queries_per_class
    new_query_weight = torch.zeros(new_num_queries, hidden_dim)
    
    for class_idx in range(num_classes):
        old_query = old_query_weight[class_idx]  # [hidden_dim]
        for k in range(k_queries_per_class):
            new_idx = class_idx * k_queries_per_class + k
            if k == 0:
                # First query is exact copy of original
                new_query_weight[new_idx] = old_query.clone()
            else:
                # Additional queries = original + small noise
                noise = torch.randn_like(old_query) * noise_frac * old_query.abs().mean()
                new_query_weight[new_idx] = old_query.clone() + noise
    
    print(f"  New queries: {new_query_weight.shape} ({num_classes} classes × {k_queries_per_class} = {new_num_queries} queries)")
    
    # Load all weights except query_embed
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    
    for key, value in pretrained_state.items():
        if key == 'query_embed.weight':
            skipped_keys.append(key)
            continue
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)
    
    # Set the new query embeddings
    model_state['query_embed.weight'] = new_query_weight
    
    model.load_state_dict(model_state)
    
    print(f"  Loaded {len(loaded_keys)} parameter tensors from checkpoint")
    print(f"  Initialized query_embed with K={k_queries_per_class} queries per class")
    if skipped_keys:
        print(f"  Skipped/reinitialized: {skipped_keys}")
    
    return model


def freeze_except_queries_and_classifier(model):
    """
    Freeze all parameters except query_embed and presence_vector (classifier).
    These are the only parts that need training for the K-query adaptation.
    """
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if 'query_embed' in name or 'presence_vector' in name:
            param.requires_grad = True
            trainable_params.append(name)
        else:
            param.requires_grad = False
            frozen_params.append(name)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = total_params - trainable_count
    
    print(f"\n  Parameter Summary:")
    print(f"  Trainable parameters: {trainable_count:,} ({trainable_count/total_params*100:.2f}%)")
    print(f"  Frozen parameters:    {frozen_count:,} ({frozen_count/total_params*100:.2f}%)")
    print(f"  Total parameters:     {total_params:,}")
    print(f"\n  Trainable layers:")
    for name in trainable_params:
        p = dict(model.named_parameters())[name]
        print(f"    - {name}: {list(p.shape)}")
    
    return model


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # Fix seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model with K queries per class
    model, criterion = build_model(args)
    model.to(device)
    model_without_ddp = model

    # Step 1: Load weights
    print("\n" + "="*60)
    if args.resume:
        print("STEP 1: Resuming from K-query checkpoint")
        print("="*60)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print(f"  Loaded checkpoint: {args.resume}")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    elif args.pretrained:
        print("STEP 1: Loading pretrained weights & initializing K queries")
        print("="*60)
        model_without_ddp = load_pretrained_and_init_k_queries(
            model_without_ddp, 
            args.pretrained, 
            args.k_queries_per_class, 
            args.noise_frac
        )
    else:
        print("STEP 1: Training from scratch (no pretrained or resume)")
        print("="*60)

    # Step 2: Freeze everything except query_embed + classifier
    print("\n" + "="*60)
    print("STEP 2: Freezing pretrained parameters")
    print("="*60)
    model_without_ddp = freeze_except_queries_and_classifier(model_without_ddp)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTrainable params: {n_parameters:,}')

    # Only optimize trainable parameters (query_embed + presence_vector)
    trainable_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Resume optimizer/scheduler/epoch if resuming
    if args.resume and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print(f"  Resumed optimizer & scheduler, starting from epoch {args.start_epoch}")

    # Build datasets
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set=args.test, args=args)

    if args.max_train_samples > 0:
        train_n = min(args.max_train_samples, len(dataset_train))
        dataset_train = torch.utils.data.Subset(dataset_train, list(range(train_n)))
        print(f"  Using subset for train: {train_n} samples")

    if args.max_eval_samples > 0:
        eval_n = min(args.max_eval_samples, len(dataset_val))
        dataset_val = torch.utils.data.Subset(dataset_val, list(range(eval_n)))
        print(f"  Using subset for eval: {eval_n} samples")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size
    data_loader_val = DataLoader(dataset_val, eval_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # Output directories
    output_dir = Path(args.output_dir)
    save_dir = output_dir / args.dataset_name / args.output_sub_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Evaluate BEFORE finetuning (baseline with replicated queries)
    # Skip when resuming - already evaluated in previous run
    if args.resume and args.start_epoch > 0:
        print("\n" + "="*60)
        print("STEP 3: Skipping pre-finetuning evaluation (resuming from checkpoint)")
        print("="*60)
        test_stats_before = {'loss': 0, 'acc1': 0, 'acc5': 0}
    else:
        print("\n" + "="*60)
        print("STEP 3: Evaluating with initialized K queries (before training)")
        print("="*60)
        test_stats_before = evaluate(model, criterion, data_loader_val, device, args.output_dir)
        print(f"\nBefore finetuning: loss={test_stats_before['loss']:.4f}, "
              f"acc1={test_stats_before['acc1']:.2f}%, acc5={test_stats_before['acc5']:.2f}%")

    if args.eval:
        if args.save_init_checkpoint:
            init_ckpt_path = save_dir / 'checkpoint_init.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'epoch': -1,
                'args': args,
            }, init_ckpt_path)
            print(f"  Saved initialized checkpoint: {init_ckpt_path}")
        return

    # Clear CUDA cache before training
    torch.cuda.empty_cache()

    # Step 4: Finetune for a few epochs
    print("\n" + "="*60)
    print(f"STEP 4: Finetuning for {args.epochs} epochs (queries + classifier only)")
    print("="*60)
    
    start_time = time.time()
    best_acc1 = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        lr_scheduler.step()

        test_stats = evaluate(model, criterion, data_loader_val, device, args.output_dir)

        # Save checkpoint
        checkpoint_paths = [save_dir / 'checkpoint.pth']
        if test_stats['acc1'] > best_acc1:
            best_acc1 = test_stats['acc1']
            checkpoint_paths.append(save_dir / 'checkpoint_best.pth')
        
        if (epoch + 1) == args.epochs:
            checkpoint_paths.append(save_dir / f'checkpoint{epoch:04}.pth')

        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        with (save_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

        print(f"  Epoch {epoch}: train_loss={train_stats['loss']:.4f}, "
              f"test_acc1={test_stats['acc1']:.2f}%, test_acc5={test_stats['acc5']:.2f}%, "
              f"best_acc1={best_acc1:.2f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Training time: {total_time_str}")
    print(f"  Before finetuning: acc1={test_stats_before['acc1']:.2f}%")
    print(f"  After finetuning:  acc1={best_acc1:.2f}%")
    print(f"  Improvement:       {best_acc1 - test_stats_before['acc1']:.2f}%")
    print(f"  Best checkpoint:   {save_dir / 'checkpoint_best.pth'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('INTR K-query finetuning', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
