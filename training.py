"""
    Resolution Matters: Impact of 3D CT Image Size on DenseNet3D CNN Performance for Lung Cancer Classification
    Author: Adem GENCER, MD
    Email: dr.ademgencer@gmail.com
"""

import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureTyped, ToTensord
)
from sklearn.model_selection import train_test_split, StratifiedKFold


# =============================
# 📌 Experimental settings
# =============================
experiment_type = "144"

# =============================



def load_data(data_dir):
    """
    Load dataset paths and labels from the given directory.
    
    Returns:
        data (list of dict): [{"ct": path, "pet": path (if available), "label": label}, ...]
        labels (numpy array): Corresponding labels.
    """
    data = []
    labels = []
    data_dir = Path(data_dir)

    for case in tqdm(sorted(data_dir.iterdir()), desc="Processing cases"):
        if not case.is_dir():
            continue  # Skip non-directory files

        try:
            # File paths
            ct_path = case / "ct.npy"
            pet_path = case / "pet.npy"  # Optional PET
            label_path = case / "label.npy"

            # Check required files
            if not ct_path.exists() or not label_path.exists():
                print(f"Skipping {case.name}: Missing required files.")
                continue

            # Load label
            label = np.load(label_path, allow_pickle=True)
            if isinstance(label, np.ndarray) and label.size == 1:
                label = label.item()  # Convert single-element array to scalar

            # Store data
            case_data = {"ct": str(ct_path), "label": label}
            if pet_path.exists():  # Include PET if available
                case_data["pet"] = str(pet_path)

            data.append(case_data)
            labels.append(label)

        except Exception as e:
            print(f"Error processing {case.name}: {e}")

    return data, np.array(labels)

def stratified_split(data, labels, test_size=0.1, val_size=0.1, random_state=42):
    """
    Splits data into stratified train, validation, and test sets.
    
    Ensures the test set is completely separate.
    """
    # First, create test set (10%)
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    # Now, split remaining data into train (80%) and validation (10%)
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=val_size / (1 - test_size),
        stratify=train_val_labels, random_state=random_state
    )

    print(f"Dataset split:\nTrain: {len(train_data)} | Validation: {len(val_data)} | Test: {len(test_data)}")
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

def stratified_kfold_cv(train_data, train_labels, n_splits=5, random_state=42):
    """
    Generates stratified 5-fold cross-validation splits.
    
    Returns a list of (train_idx, val_idx) tuples.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_splits = list(skf.split(train_data, train_labels))

    print(f"Generated {n_splits}-Fold CV splits")
    return fold_splits

# Define MONAI Transforms
def get_transforms():
    """
    Returns MONAI transformations for pre-processing and augmentation.
    """
    return Compose([
        LoadImaged(keys=["ct"], reader="numpyreader"),
        # EnsureTyped(keys=["ct", "label"]),
        ToTensord(keys=["ct", "label"]),
    ])

# Load dataset
data_directory = f"./npy_lung_{experiment_type}"
data, labels = load_data(data_directory)

# Split data
train_data, val_data, test_data, train_labels, val_labels, test_labels = stratified_split(data, labels)

# Create Datasets
train_ds = CacheDataset(train_data, transform=get_transforms(), cache_rate=0.1, num_workers=4)
val_ds = CacheDataset(val_data, transform=get_transforms(), cache_rate=0.1, num_workers=4)
test_ds = Dataset(test_data, transform=get_transforms())  # No caching for test data

# Create DataLoaders
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)


# K-Fold Cross Validation (Exclude Test Set)
fold_splits = stratified_kfold_cv(train_data, train_labels)

# Print Fold Splits
for fold, (train_idx, val_idx) in enumerate(fold_splits):
    print(f"Fold {fold + 1}: Train size: {len(train_idx)}, Validation size: {len(val_idx)}")

# Example Usage
for batch in train_loader:
    print(batch["ct"].shape, batch["label"])
    break  # Print first batch only


import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import wandb
from monai.networks.nets import DenseNet121
from monai.data import DataLoader, CacheDataset, Dataset
from monai.transforms import Compose, LoadImaged, ToTensord, EnsureType, Activations, AsDiscrete
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW


# =============================
# 📌 Initialize Weights & Biases
# =============================
wandb.init(
    project="3D-DenseNet121-Lung-CT",
    # name="densenet3d_ct_4class",
    config={
        "model": "DenseNet121-3D",
        "modality": "CT (1 channel)",
        "experiment": "Size comparison",
        "experiment_type": experiment_type,
        "loss_function": "Weighted Cross Entropy",
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "workers": 4,
        "batch_size": 4,    
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "num_classes": 4,
        "input_size": 1,
        "drop_out": 0.1,
        "early_stopping_patience": 50,
        "max_epochs": 1000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
)

# =============================
# 📌 Setup GPU
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.config["gpu"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
print(f"Using device: {device}")


# =============================
# 📌 Model Definition
# =============================
model = DenseNet121(
    spatial_dims=3,
    in_channels=1,
    out_channels=4,
    dropout_prob=0.1  # Add dropout to improve generalization
).to(device)

# =============================
# 📌 Class Weighted Cross-Entropy Loss
# =============================
class_counts = np.bincount(train_labels)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# =============================
# 📌 Optimizer & Scheduler
# =============================
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)


# =============================
# 📌 Training loop
# =============================
best_val_loss = float("inf")
early_stop_counter = 0
save_path = f"./saved_models_{experiment_type}"
os.makedirs(save_path, exist_ok=True)

start_time = time.time()
for epoch in range(1000):
    model.train()
    train_loss, train_correct, total_train = 0.0, 0, 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{1000} [Training]", leave=False)

    for batch in train_pbar:
        inputs, labels = batch["ct"].to(device), batch["label"].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_train += labels.size(0)

        train_pbar.set_postfix(loss=loss.item())

    train_acc = train_correct / total_train
    train_loss /= total_train

    # Validation
    model.eval()
    val_loss, val_correct, total_val = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["ct"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_val += labels.size(0)

    val_acc = val_correct / total_val
    val_loss /= total_val

    # Logging to WandB
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

    # Reduce LR on Plateau
    scheduler.step(val_loss)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(save_path, f"best_model_{experiment_type}.pth"))
        wandb.save(os.path.join(save_path, f"best_model_{experiment_type}.pth"))
    else:
        early_stop_counter += 1
        if early_stop_counter >= 50:
            print("Early stopping triggered.")
            break

end_time = time.time()
wandb.log({"total_training_time": end_time - start_time})

# =============================
# 📌 Testing
# =============================
model.load_state_dict(torch.load(os.path.join(save_path, f"best_model_{experiment_type}.pth")))
model.eval()

y_true, y_pred, y_probs = [], [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        inputs, labels = batch["ct"].to(device), batch["label"].to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds)
        y_probs.extend(probs)

# =============================
# 📌 Compute metrics
# =============================
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

test_metrics = {
    "test_acc": accuracy_score(y_true, y_pred),
    "test_auc": roc_auc_score(y_true, y_probs, multi_class='ovr'),
    "test_f1": f1_score(y_true, y_pred, average='weighted'),
    "test_precision": precision_score(y_true, y_pred, average='weighted'),
    "test_recall": recall_score(y_true, y_pred, average='weighted')
}

wandb.log(test_metrics)
print("Test Metrics:", test_metrics)
wandb.finish()
