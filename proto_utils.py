import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
from utils import *
from agents import *
import time
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from copy import deepcopy
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import resnet18
import random
import math
from ov_utils import *

def update_fc_with_prototypes(
    model,
    attack_loader,
    unlearn_classes,
    num_samples_per_class=5,
    metric='cosine',
    normalize_proto=True,
    alpha=0.5,
    device='cuda'
):
    model = model.to(device)
    model.eval()

    feature_dict = {c: [] for c in unlearn_classes}
    counts       = {c: 0  for c in unlearn_classes}

    with torch.no_grad():
        for images, labels in attack_loader:
            images, labels = images.to(device), labels.to(device)
            feats = model.avg_pool(model.features(images))
            feats = feats.view(feats.size(0), -1)  # (B, D)

            for feat, lbl in zip(feats, labels):
                c = lbl.item()
                if c in counts and counts[c] < num_samples_per_class:
                    feature_dict[c].append(feat.clone())
                    counts[c] += 1

            if all(counts[c] >= num_samples_per_class for c in unlearn_classes):
                break

    proto_dict = {}
    for c, flist in feature_dict.items():
        proto = torch.stack(flist, dim=0).mean(dim=0)
        if normalize_proto:
            proto = F.normalize(proto, p=2, dim=0)
        proto_dict[c] = proto.to(device)

    orig_w = model.fc.weight.data.clone()
    orig_b = model.fc.bias.data.clone()

    for c, proto in proto_dict.items():
        if metric == 'l2':
            w_proto = 2.0 * proto
            b_proto = - proto.pow(2).sum()
        elif metric == 'cosine':
            w_proto = proto
            b_proto = torch.tensor(0.0, device=device)

        w_new = alpha * w_proto + (1 - alpha) * orig_w[c]
        b_new = alpha * b_proto + (1 - alpha) * orig_b[c]

        model.fc.weight.data[c] = w_new
        model.fc.bias.data[c]   = b_new

    return model

def create_attack_loaders(
    test_set,
    n_percent,
    unlearn_classes,
    remain_classes,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    seed=None
):

    total = len(test_set)
    k = int(total * (n_percent / 100.0))
    if seed is not None:
        random.seed(seed)
    all_indices = list(range(total))
    sampled_indices = random.sample(all_indices, k)
    attack_subset = Subset(test_set, sampled_indices)
    targets = test_set.targets
    unlearn_idx = [idx for idx in sampled_indices if targets[idx] in unlearn_classes]
    remain_idx  = [idx for idx in sampled_indices if targets[idx] in remain_classes]
    unlearn_subset = Subset(test_set, unlearn_idx)
    remain_subset  = Subset(test_set, remain_idx)

    unlearn_attack_loader = DataLoader(
        unlearn_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    remain_attack_loader = DataLoader(
        remain_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    unlearn_count = len(unlearn_idx)
    remain_count  = len(remain_idx)
    print(f"Attack subset size: {len(sampled_indices)}")
    print(f"  → Unlearn class samples: {unlearn_count}")
    print(f"  → Remain class samples: {remain_count}")

    counts = {'unlearn': unlearn_count, 'remain': remain_count}
    return attack_subset, unlearn_attack_loader, remain_attack_loader, counts

def extract_features(model, x, device='cuda'):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        feats = model.features(x)
        feats = model.avg_pool(feats)
        feats = feats.view(feats.size(0), -1)
    return feats