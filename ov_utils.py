import os
from utils import *
from agents import *
import time
import torch
import torch.nn as nn
import tqdm
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
import random
import math
from torch.utils.data import TensorDataset, DataLoader

def mask_target_soft(teacher_probs, labels):
    teacher_probs_mod = teacher_probs.clone()
    for i, label in enumerate(labels):
        teacher_probs_mod[i, label] = 0.0
    row_sum = teacher_probs_mod.sum(dim=1, keepdim=True)
    teacher_probs_mod = teacher_probs_mod / (row_sum + 1e-10)
    return teacher_probs_mod

def mask_full_target_soft(teacher_probs, labels):
    teacher_probs_mod = teacher_probs.clone()
    for i in range(len(teacher_probs_mod)):
        for label in labels:
            teacher_probs_mod[i, label] = 0.0
    row_sum = teacher_probs_mod.sum(dim=1, keepdim=True)
    teacher_probs_mod = teacher_probs_mod / (row_sum + 1e-10)
    return teacher_probs_mod

def get_features(model, x):
    feat = model.features(x)
    feat = model.avg_pool(feat)
    feat = feat.view(feat.size(0), -1)
    return feat

def get_vit_features(model, x):
    feats = model.forward_features(x)
    if feats.dim() == 3:
        feats = feats[:, 0, :]
    return feats

def dispersion_loss(features, labels, metric = "cosine"):
    N = features.size(0)
    eye_mask = torch.eye(N, dtype=torch.bool, device=features.device)
    same_class = labels.unsqueeze(1) == labels.unsqueeze(0)
    same_class &= ~eye_mask
    num_pairs = same_class.sum().float().clamp(min=1)

    if metric == "cosine":
        feats = F.normalize(features, p=2, dim=1)
        sim_matrix = feats @ feats.t()                # (N, N)
        loss_disp = sim_matrix[same_class].sum() / num_pairs
    else:
        dist_matrix = torch.cdist(features, features, p=2)
        loss_disp = dist_matrix[same_class].sum() / num_pairs

    return loss_disp

def compute_dispersion_loss(model, data_loader, metric = "cosine"):

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            feats = get_features(model, inputs)
            all_features.append(feats)
            all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    loss_disp = dispersion_loss(all_features, all_labels, metric)
    
    return loss_disp

def compute_divergence_adjusted(model_F, model_Fu, adv_loader, u_labels, mode="JS", device='cuda'):
    model_F.eval()
    model_Fu.eval()
    
    total_div = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _ in adv_loader:
            images = images.to(device)

            logits_F = model_F(images)
            logits_Fu = model_Fu(images)
            probs_F = F.softmax(logits_F, dim=1)
            probs_Fu = F.softmax(logits_Fu, dim=1)

            probs_F_adj = mask_full_target_soft(probs_F, u_labels)
            
            if mode.upper() == "JS":
                M = (probs_F_adj + probs_Fu) / 2.0
                kl1 = F.kl_div(torch.log(M + 1e-12), probs_F_adj, reduction='batchmean')
                kl2 = F.kl_div(torch.log(M + 1e-12), probs_Fu, reduction='batchmean')
                divergence = 0.5 * (kl1 + kl2)
            
            batch_size = images.size(0)
            total_div += divergence.item() * batch_size
            total_samples += batch_size
            
    avg_divergence = total_div / total_samples
    return avg_divergence


def compare_unlearning_methods(model_F, models_Fu, adv_loader, u_labels, mode, device='cuda'):
    results = {}
    for key, model_fu in models_Fu.items():
        score = compute_divergence_adjusted(model_F, model_fu, adv_loader, u_labels, mode, device=device)
        results[key] = score
    return results

def create_adversarial_loader_pgd_multi(
    model_F,
    clean_loader,
    epsilon=0.03,
    step_size=0.01,
    num_steps=3,
    num_adv_per_sample=1,
    device='cuda'
):
    model_F.eval().to(device)
    
    adv_examples_list = []
    dummy_labels_list = []
    
    for images, labels in clean_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model_F(images)
            top2 = logits.argsort(dim=1, descending=True)[:, :2]  # shape: (B, 2)
    
        target_labels = []
        for i in range(labels.size(0)):
            top1_label = top2[i, 0]
            top2_label = top2[i, 1]
            if top1_label.item() == labels[i].item():
                target_labels.append(top2_label.item())
            else:
                target_labels.append(top1_label.item())
                
        target_labels = torch.LongTensor(target_labels).to(device)  # shape: (B,)
        images_rep = images.repeat_interleave(num_adv_per_sample, dim=0)
        target_labels_rep = target_labels.repeat_interleave(num_adv_per_sample, dim=0)
        noise = torch.empty_like(images_rep).uniform_(-epsilon, epsilon)
        x_adv = images_rep + noise
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(num_steps):
            x_adv.requires_grad_()
            logits_adv = model_F(x_adv)
            loss = F.cross_entropy(logits_adv, target_labels_rep)
            model_F.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + step_size * grad_sign
                x_adv = torch.max(torch.min(x_adv, images_rep + epsilon), images_rep - epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
                

        adv_examples_list.append(x_adv.detach().cpu())
        dummy_labels_list.append(labels.repeat_interleave(num_adv_per_sample, dim=0).cpu())

    adv_examples = torch.cat(adv_examples_list, dim=0)
    dummy_labels = torch.cat(dummy_labels_list, dim=0)
    adv_dataset = TensorDataset(adv_examples, dummy_labels)
    adv_loader = DataLoader(adv_dataset, batch_size=clean_loader.batch_size, shuffle=False)
    
    return adv_loader


def create_adversarial_loader_gaussian(
    model_F,
    clean_loader,
    noise_std=0.01,
    num_adv_per_sample=1,
    device='cuda'
):
    model_F.eval().to(device)
    
    adv_examples_list = []
    dummy_labels_list = []
    
    for images, labels in clean_loader:
        images = images.to(device)
        labels = labels.to(device)
        images_rep = images.repeat_interleave(num_adv_per_sample, dim=0)
        noise = torch.randn_like(images_rep) * noise_std
        x_adv = images_rep + noise
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        adv_examples_list.append(x_adv.detach().cpu())
        dummy_labels_list.append(labels.repeat_interleave(num_adv_per_sample, dim=0).cpu())

    adv_examples = torch.cat(adv_examples_list, dim=0)
    dummy_labels = torch.cat(dummy_labels_list, dim=0)
    adv_dataset = TensorDataset(adv_examples, dummy_labels)
    adv_loader = DataLoader(adv_dataset, batch_size=clean_loader.batch_size, shuffle=False)
    
    return adv_loader
