# Install other requirements
!pip install nibabel
!pip install wandb
!pip install wandb --upgrade
!pip install monai
!pip install monai[all]
import monai
print("-------- Installation completed! ----------")


# =================================================
# Import Libraries
# =================================================
import os
import json
import numpy as np
import nibabel
from tqdm import tqdm
from random import shuffle
from collections import Counter
import wandb

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Spacingd, Orientationd, ScaleIntensityRanged, CropForegroundd, Resized,
    RandFlipd, RandRotate90d, RandAdjustContrastd, RandGaussianNoised, 
    ToTensord
)
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from monai.networks.nets import DenseNet121

# =================================================
# Settings
# =================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
data_directory = "./npy_datasetv2_96"
sample_size = 1000
split_size = [0.8, 0.1, 0.1]
exclude_classes = []
# label_mapping = {0: "None", 1: "Adenokarsinom", 2: "Squamoz", 3: "SmallCell"}
label_mapping = {0: "None", 1: "Adenokarsinom", 2: "Squamoz", 3: "SmallCell", 4: "Benign"}
# label_mapping = {0: "Adenokarsinom", 1: "Squamoz", 2: "SmallCell"}
random_seed = 44
set_determinism(seed=random_seed)
batch_size = 12
num_workers = 6
max_epochs = 1000

# Preprocess settings
"""
--- Configured in preprocessing stage ---
voxel_spacing = (1.0, 1.0, 1.0)
image_size = (128, 128, 128)
hu_range = {"a_min": -1000, "a_max": 200}  # Hounsfield Unit range for CT
suv_range = {"a_min": 0, "a_max": 20}      # Standard Uptake Value range for PET (NOT SUV_MAX ! Activity (MBq): 5.286 For every SUV Value.
"""

# =================================================
# Load and Prepare Dataset
# =================================================
def load_data(data_dir):
    data = []
    # label_mapping = {1: 0, 2: 1, 3: 2}  # Map old labels to new labels
    
    for case in tqdm(sorted(os.listdir(data_dir)), desc="Processing cases"):
        case_path = os.path.join(data_dir, case)
        if not os.path.isdir(case_path):
            continue
        
        ct_path = os.path.join(case_path, "ct.npy")
        pet_path = os.path.join(case_path, "pet.npy")
        label_path = os.path.join(case_path, "label.npy")
        
        try:
            # Load the label from label.npy
            label = np.load(label_path)
            if isinstance(label, np.ndarray) and label.size == 1:
                label = label.item()

            
            if label not in exclude_classes:  # Exclude specified classes
                new_label = label_mapping.get(label, None)

                # if new_label is not None:  # Ensure valid mapping
                #    data.append({"ct": ct_path, "pet": pet_path, "label": new_label})
                # else:
                #    print(f"❌ Warning: Unexpected label {label} in {case}")
                data.append({"ct": ct_path, "pet": pet_path, "label": label})
        except Exception as e:
            print(f"Error processing {case}: {e}")
    return data

dataset = load_data(data_directory)
shuffle(dataset)
dataset = dataset[:sample_size]
labels = [item['label'] for item in dataset]
# print("Unique labels in dataset:", np.unique(labels))
class_counts = Counter(labels)
num_classes = len(class_counts)

# =================================================
# Split Dataset
# =================================================
train_size, val_size, test_size = [x / sum(split_size) for x in split_size]
if any(count < 2 for count in class_counts.values()):
    print("Warning: Rare classes detected. Using random split.")
    train_val, test_data = train_test_split(dataset, test_size=test_size, random_state=random_seed)
    train_data, val_data = train_test_split(train_val, test_size=val_size / (train_size + val_size), random_state=random_seed)
else:
    train_val, test_data = train_test_split(dataset, test_size=test_size, stratify=labels, random_state=random_seed)
    train_labels = [item['label'] for item in train_val]
    train_data, val_data = train_test_split(train_val, test_size=val_size / (train_size + val_size),
                                            stratify=train_labels, random_state=random_seed)

# =================================================
# Define Transforms and Dataloaders
"""
    LoadImaged(keys=["ct", "pet"], reader="PydicomReader"),
    EnsureChannelFirstd(keys=["ct", "pet"]),
    Spacingd(keys=["ct", "pet"], pixdim=voxel_spacing, mode=("trilinear", "trilinear")),
    Orientationd(keys=["ct", "pet"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["ct"], a_min=hu_range["a_min"], a_max=hu_range["a_max"], b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=["pet"], a_min=suv_range["a_min"], a_max=suv_range["a_max"], b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["ct", "pet"], source_key="ct", allow_smaller=True),
    Resized(keys=["ct", "pet"], spatial_size=image_size, mode=("trilinear", "trilinear")),
    EnsureTyped(keys=["ct", "pet"]),
"""
# =================================================


train_transforms = Compose([
    LoadImaged(keys=["ct", "pet"], reader="numpyreader"),
    # RandFlipd(keys=["ct", "pet"], prob=0.5, spatial_axis=0),
    # RandRotate90d(keys=["ct", "pet"], prob=0.5, max_k=3),
    # RandAdjustContrastd(keys=["ct", "pet"], prob=0.2, gamma=(0.95, 1.05)),
    # RandGaussianNoised(keys=["ct", "pet"], prob=0.2),
    ToTensord(keys=["ct", "pet", "label"]),
])

val_test_transforms = Compose([
    LoadImaged(keys=["ct", "pet"], reader="numpyreader"),
    ToTensord(keys=["ct", "pet", "label"]),
])

train_dataset = Dataset(data=train_data, transform=train_transforms)
val_dataset = Dataset(data=val_data, transform=val_test_transforms)
test_dataset = Dataset(data=test_data, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# =================================================
# Model and Training Loop with Early Stopping
# =================================================
model = DenseNet121(spatial_dims=3, in_channels=2, out_channels=num_classes, dropout_prob=0.1).to(device)

# Class weights for imbalance handling
class_weights = torch.tensor([len(dataset) / class_counts.get(i, 1e-6) for i in range(num_classes)], dtype=torch.float32).to(device)
class_weights /= class_weights.sum()  # Optional: Normalize class weights
loss_function = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

# Early Stopping Parameters
patience = 20  # Number of epochs to wait before stopping
best_val_loss = float("inf")
early_stop_counter = 0
early_stop_triggered = False  # To check if early stopping is triggered

# Initialize WandB
wandb.init(project="lung-cancer-classification", config={
    "model": "DenseNet121",
    "exp": "size: 144, class: except 0, mod: ct-pet",
    "epochs": max_epochs,
    "batch_size": train_loader.batch_size,
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-3
})
wandb.watch(model, log="all", log_freq=100)

for epoch in range(max_epochs):
    # Training
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
        inputs = torch.cat([batch["ct"], batch["pet"]], dim=1).to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = correct_train / total_train

    # Log training metrics to WandB
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss / len(train_loader),
        "train_accuracy": train_accuracy,
        "learning_rate": optimizer.param_groups[0]["lr"]
    })

    # Validation
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
            inputs = torch.cat([batch["ct"], batch["pet"]], dim=1).to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Accumulate loss
            val_loss += loss.item() * inputs.size(0)  # Scale loss by batch size
            preds = outputs.argmax(dim=1)

            # Accumulate predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    # Compute average metrics
    val_loss = val_loss / total_val
    val_accuracy = correct_val / total_val

    # Log validation metrics
    wandb.log({
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    # Save the best model to WandB
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0  # Reset early stopping counter
        # Save model weights to WandB
        torch.save(model.state_dict(), "best_model.pth")
        wandb.save("best_model.pth")
        # print(f"Epoch {epoch + 1}: New best model saved with val_loss {val_loss:.4f}")
    else:
        early_stop_counter += 1

    # Log confusion matrix every 10 epochs
    if (epoch + 1) % 10 == 0 or early_stop_triggered:
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        class_accuracy = cm.diagonal() / cm.sum(axis=1)

        # Log class-wise accuracies
        for i, acc in enumerate(class_accuracy):
            wandb.log({f"class_{i}_accuracy": acc})

        # Plot and log confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_mapping.values())
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
        plt.close(fig)
        wandb.log({"confusion_matrix": wandb.Image(fig)})

    # Adjust learning rate with scheduler
    scheduler.step(val_loss)

    # Print epoch summary
    # print(f"Epoch {epoch + 1}: Train Loss {train_loss / len(train_loader):.4f}, "
    #      f"Train Acc {train_accuracy:.4f}, Val Loss {val_loss:.4f}, "
    #      f"Val Acc {val_accuracy:.4f}")

    # Early stopping check
    if early_stop_counter >= patience:
        early_stop_triggered = True
        print(f"Early stopping triggered at epoch {epoch + 1}. Best val_loss: {best_val_loss:.4f}")
        break

# Log final confusion matrix after early stopping
if early_stop_triggered:
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # Log class-wise accuracies
    for i, acc in enumerate(class_accuracy):
        wandb.log({f"class_{i}_accuracy": acc})

    # Plot and log confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_mapping.values())
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Final Confusion Matrix")
    plt.close(fig)
    wandb.log({"final_confusion_matrix": wandb.Image(fig)})

# Finish the WandB run
wandb.finish()
