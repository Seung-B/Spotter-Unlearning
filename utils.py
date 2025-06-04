import torch
import numpy as np
import umap
from tqdm import tqdm
import torchvision
from torchvision import transforms
from copy import deepcopy
import numpy as np
import random
import os
import torchvision


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate(model, loader, loader_name, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    print(f"[{loader_name}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}% ({correct}/{total})")