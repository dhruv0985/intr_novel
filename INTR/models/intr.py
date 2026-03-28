# ------------------------------------------------------------------------
# INTR
# Copyright (c) 2023 Imageomics Paul. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
INTR model and loss.
"""
import torch
from torch import nn
import torch.nn.functional as F

import random
from .backbone import build_backbone
from .transformer import build_transformer
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       )

class INTR(nn.Module):
    """ This is the INTR module that performs explainable image classification """
    def __init__(self, args, backbone, transformer, num_classes, k_queries_per_class=1):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py (no pos_embed in decoder)
            num_classes: number of classes in the dataset
            k_queries_per_class: number of queries per class (default: 1)
        """
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.k_queries_per_class = k_queries_per_class
        self.num_queries = num_classes * k_queries_per_class  # Total number of queries
        self.query_aggregation = getattr(args, 'query_aggregation', 'max')
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # INTR classification head presence vector
        self.presence_vector = nn.Linear(hidden_dim, 1)

        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor):

        """  The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]

            It returns the following elements:
               - "out": it is a dictnary which currently contains all logit values for for all queries.
                                Shape= [batch_size x num_queries x 1]
               - "encoder_output": it is the output of the transformer encoder which is basically feature map. 
                                Shape= [batch_size x num_features x height x weight]
               - "hs": it is the output of the transformer decoder. These are learned class specific queries. 
                                Shape= [dec_layers x batch_size x num_queries x num_features]
               - "attention_scores": it is attention weight corresponding to each pixel in the encoder  for all heads. 
                                Shape= [dec_layers x batch_size x num_heads x num_queries x height*weight]
               - "avg_attention_scores": it is attention weight corresponding to each pixel in the encoder for avg of all heads. 
                                Shape= [dec_layers x batch_size x num_queries x height*weight]

        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()

        assert mask is not None
        hs, encoder_output, attention_scores, avg_attention_scores = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        query_logits = self.presence_vector(hs[-1]).squeeze(dim=-1)  # [batch_size, num_queries]
        
        # Aggregate K queries per class
        batch_size = query_logits.shape[0]
        # Reshape: [batch_size, num_classes, k_queries_per_class]
        query_logits_reshaped = query_logits.view(batch_size, self.num_classes, self.k_queries_per_class)
        
        # Aggregate across k_queries_per_class dimension
        if self.query_aggregation == 'max':
            class_logits, _ = torch.max(query_logits_reshaped, dim=2)  # [batch_size, num_classes]
        elif self.query_aggregation == 'mean':
            class_logits = torch.mean(query_logits_reshaped, dim=2)  # [batch_size, num_classes]
        elif self.query_aggregation == 'sum':
            class_logits = torch.sum(query_logits_reshaped, dim=2)  # [batch_size, num_classes]
        else:
            class_logits, _ = torch.max(query_logits_reshaped, dim=2)  # default to max
        
        out = {
            'query_logits': class_logits,  # [batch_size, num_classes] - used for loss/accuracy
            'all_query_logits': query_logits,  # [batch_size, num_queries] - for visualization
            'query_logits_per_class': query_logits_reshaped  # [batch_size, num_classes, k] - for analysis
        }

        return out, encoder_output, hs, attention_scores, avg_attention_scores


class SetCriterion(nn.Module):
    """ This class computes the loss for INTR.
        INTR uses only one type of loss i.e., cross entropy loss.
    """
    def __init__(self, args,  model): # weight_dict, losses,
        """ Create the criterion.
        """
        super().__init__()
        self.args = args
        self.model = model

    def get_loss(self, outputs, targets, model):
        """ CE Classification loss
        targets dicts must contain the key "image_label".
        """
        assert 'query_logits' in outputs
        query_logits = outputs['query_logits']
        device = query_logits.device

        target_classes = torch.cat([t['image_label'] for t in targets]) 
        
        criterion = torch.nn.CrossEntropyLoss()
        classification_loss=criterion(query_logits, target_classes)

        losses = {'CE_loss': classification_loss}
        return losses

    def forward(self, outputs, targets, model):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format.
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied. Here we have used only CE loss.
        """
        losses = {}
        losses.update(self.get_loss(outputs, targets, model))
        return losses


def build(args):
    """
    In INTR, each query is responsible for learning class specific information.
    With k_queries_per_class, we have K queries per class.
    Total queries = num_classes * k_queries_per_class
    """

    # Set number of classes based on dataset
    if args.dataset_name == 'cub' or 'CUB' in args.dataset_name:
        num_classes = 200
    # elif args.dataset_name== 'bird525':
    #     num_classes=525
    # elif args.dataset_name== 'fish':
    #     num_classes=183
    # elif args.dataset_name== 'dog':
    #     num_classes=120
    # elif args.dataset_name== 'butterfly':
    #     num_classes=65
    # elif args.dataset_name== 'pet':
    #     num_classes=37
    # elif args.dataset_name== 'car':
    #     num_classes=196
    # elif args.dataset_name== 'craft':
    #     num_classes=100
    else:
        num_classes = args.num_queries  # fallback to num_queries arg
    
    # Get k_queries_per_class from args (default to 1 for backward compatibility)
    k_queries_per_class = getattr(args, 'k_queries_per_class', 1)
    
    # Update num_queries to be total queries
    args.num_queries = num_classes * k_queries_per_class

    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = INTR(
        args,
        backbone,
        transformer,
        num_classes=num_classes,
        k_queries_per_class=k_queries_per_class,
        )

    criterion = SetCriterion(args, model=model)
    criterion.to(device)

    return model, criterion
