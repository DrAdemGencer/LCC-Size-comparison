"""
    Resolution Matters: Impact of 3D CT Image Size on DenseNet3D CNN Performance for Lung Cancer Classification
    Author: Adem GENCER, MD
    Email: dr.ademgencer@gmail.com
"""

# =================================================
# 1. Libraries
# =================================================

# From system
import os
import json
import numpy as np
import shutil
from tqdm import tqdm
from random import shuffle
from concurrent.futures import ThreadPoolExecutor

# From pytorch
import torch

# From monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, Resized, EnsureTyped,  
)
from monai.transforms import MapTransform

# =================================================
# 1. Settings
# =================================================

# Device settings
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Directory settings
data_directory = "./lung_roi"
logs_directory = "./logs"
pdata_directory = "./npy_lung_32"

# Data settings
sample_size = 1000
exclude_classes = []

# Preprocess settings
voxel_spacing = (2.0, 2.0, 2.0)
image_size = (32, 32, 32)
hu_range = {"a_min": -1000, "a_max": 200}  # Hounsfield Unit range for CT
suv_range = {"a_min": 0, "a_max": 15}      # Standard Uptake Value range for PET (NOT SUV_MAX ! Activity (MBq): 5.286 For every SUV Value.

# Processing settings
num_workers = 6 # Utilize 8 CPUs for processing

# =================================================
# Load and Prepare Dataset
# =================================================
def load_data(data_dir):
    data = []
    for case in tqdm(sorted(os.listdir(data_dir)), desc="Processing cases"):
        case_path = os.path.join(data_dir, case)
        if not os.path.isdir(case_path):
            continue
        
        ct_path = os.path.join(case_path, "ct.nii.gz")
        pet_path = os.path.join(case_path, "pet.nii.gz")
        label_path = os.path.join(case_path, "label.json")
        
        try:
            # Check if the required files exist
            if not (os.path.exists(ct_path) and os.path.exists(pet_path) and os.path.exists(label_path)):
                print(f"Warning: Missing files for case {case}. Skipping...")
                continue

            # Load the label from label.json
            with open(label_path, "r") as label_file:
                label_data = json.load(label_file)
            
            label = label_data.get("class")  # Assuming label.json contains {"class": <int>}
            
            if label not in exclude_classes:  # Exclude specified classes
                data.append({"ct": ct_path, "pet": pet_path, "label": label})
        except Exception as e:
            print(f"Error processing {case}: {e}")
            
    return data


dataset = load_data(data_directory)
shuffle(dataset)
dataset = dataset[:sample_size]

print(f"Total case in dataset directory: {len(dataset)}")

# =================================================
# 1. Preprocessing dataset
# =================================================


# Define preprocessing transforms
preprocessing_transforms = Compose([
    LoadImaged(keys=["ct", "pet"], reader="PydicomReader"),
    EnsureChannelFirstd(keys=["ct", "pet"]),
    Spacingd(keys=["ct", "pet"], pixdim=voxel_spacing, mode=("trilinear", "trilinear")),
    # Orientationd(keys=["ct", "pet"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["ct"], a_min=hu_range["a_min"], a_max=hu_range["a_max"], b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=["pet"], a_min=suv_range["a_min"], a_max=suv_range["a_max"], b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["ct", "pet"], source_key="ct", allow_smaller=True),
    Resized(keys=["ct", "pet"], spatial_size=image_size, mode=("trilinear", "trilinear")),
    EnsureTyped(keys=["ct", "pet"]),
])

# Overwrite folder if it exists
if os.path.exists(pdata_directory):
    shutil.rmtree(pdata_directory)  # Remove the existing directory

# Create the directory
os.makedirs(pdata_directory, exist_ok=True)

# Define a single preprocessing function
def preprocess_and_save(case, pdata_directory, preprocessing_transforms):
    try:
        # Apply preprocessing
        # print(os.path.basename(os.path.dirname(case["ct"])))
        data = preprocessing_transforms(case)
        
        # Extract case ID from the parent folder of the CT path
        case_id = os.path.basename(os.path.dirname(case["ct"]))  # Example: "case0001"

        # Create a directory for this case
        case_dir = os.path.join(pdata_directory, case_id)
        os.makedirs(case_dir, exist_ok=True)

        # Generate file paths
        ct_path = os.path.join(case_dir, "ct.npy")
        pet_path = os.path.join(case_dir, "pet.npy")
        label_path = os.path.join(case_dir, "label.npy") if "label" in case else None

        # Save data as .npy
        np.save(ct_path, data["ct"])
        np.save(pet_path, data["pet"])
        if label_path:
            np.save(label_path, np.array(case["label"]))

        return f"Processed {case_id}"
    except Exception as e:
        return f"Failed {case['ct']}  with error: {str(e)}\nDetails: {error_details}"

    
# ------ Process -----
    
# Initialize error logs
error_logs = []

# Function to process a single case
def process_case(case):
    return preprocess_and_save(case, pdata_directory, preprocessing_transforms)

# Process cases concurrently with a progress bar
with ThreadPoolExecutor(max_workers=num_workers) as executor:  # Utilize all 8 CPUs
    # Use tqdm to wrap the executor's map function
    results = list(tqdm(executor.map(process_case, dataset), total=len(dataset), desc="Processing Cases"))

# Collect error messages from results
error_logs = [result for result in results if result.startswith("Failed")]

# Display error logs after all cases
if error_logs:
    print("\nError Logs:")
    for error in error_logs:
        print(error)
else:
    print("\nAll cases processed successfully.")
