"""
Steps 1 & 2 COMP 257 Project

This script performs the complete data preparation pipeline for the UMIST
face dataset (umist_cropped.mat), including:

- Loading and exploring the UMIST dataset structure
- Extracting face images and corresponding person labels
- Flattening images into feature vectors
- Converting features and labels into a Pandas DataFrame
- Performing stratified train / validation / test splitting to preserve
  class balance across all subsets
- Normalizing features using StandardScaler (fit on training data only)
- Visualizing sample images with correct labels
- Visualizing class distribution per split
- Saving processed data splits, scaler, and metadata for use in later
  stages of the project (dimensionality reduction, clustering, and
  supervised learning)
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import json
import joblib

plt.style.use("ggplot")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "umist_cropped.mat")
RANDOM_STATE = 42


def load_umist_mat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(path)

    keys = [k for k in mat.keys() if not k.startswith("__")]
    print("MAT keys:", keys)

    if "facedat" not in mat:
        raise KeyError("Key 'facedat' not found in MAT file.")

    facedat = mat["facedat"]

    # Normalize facedat into a 1D sequence of subject arrays
    if facedat.ndim == 2:
        if facedat.shape[0] == 1:
            subjects = facedat[0]
        elif facedat.shape[1] == 1:
            subjects = facedat[:, 0]
        else:
            subjects = facedat.ravel()
    else:
        subjects = facedat

    images_list = []
    labels_list = []

    # Iterate over each subject and collect their images with a numeric label.
    for person_idx, subject_data in enumerate(subjects):
        subject_arr = np.array(subject_data)

        if subject_arr.ndim == 2:
            # subject_data is supposed to be a 3D array
            subject_arr = subject_arr[:, :, np.newaxis]

        if subject_arr.ndim != 3:
            raise ValueError(f"Wrong shape for subject {person_idx}: {subject_arr.shape}.")

        h, w, num_images = subject_arr.shape

        # Loop through each image for this subject and append to the lists.
        for j in range(num_images):
            img = subject_arr[:, :, j]
            images_list.append(img.astype(np.float32))
            labels_list.append(person_idx)

    images = np.stack(images_list, axis=0)
    labels = np.array(labels_list, dtype=np.int64)

    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of subjects: {len(subjects)}")

    return images, labels


# Flatten images into vectors of shape
def flatten_images(images: np.ndarray) -> np.ndarray:
    if images.ndim != 3:
        raise ValueError("Image shape is incorrect; expected (N,H,W).")

    n_samples, h, w = images.shape
    X = images.reshape(n_samples, h * w)
    print(f"Flattened images to shape: {X.shape}")
    return X


# Convert to DataFrame (rubric)
def to_dataframe(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(X)
    df["label"] = y
    return df


# Plot sample images with labels (rubric)
def plot_sample_images(images, labels, n=12, save_path=None):
    rng = np.random.RandomState(RANDOM_STATE)
    n = min(n, len(images))
    idx = rng.choice(len(images), size=n, replace=False)

    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4))
    axes = np.array(axes).ravel()

    for k, i in enumerate(idx):
        axes[k].imshow(images[i], cmap="gray")
        axes[k].set_title(f"ID {labels[i]}")
        axes[k].axis("off")

    for j in range(k + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Sample UMIST Images with Labels")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved sample images plot to {save_path}")
    else:
        plt.show()


# Safety check for stratified splitting
def check_min_samples_per_class(y, min_required=3):
    counts = pd.Series(y).value_counts()
    if (counts < min_required).any():
        bad = counts[counts < min_required]
        raise ValueError(
            "Some classes have too few samples for stratified split.\n"
            f"Minimum required per class: {min_required}\n"
            f"Problem classes:\n{bad}"
        )


def split_and_scale(X: np.ndarray, y: np.ndarray):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print("Split sizes:")
    print("  Train:", X_train.shape[0])
    print("  Val  :", X_val.shape[0])
    print("  Test :", X_test.shape[0])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def plot_class_distribution(y_train, y_val, y_test, save_path=None):
    splits = [("Train", y_train), ("Validation", y_val), ("Test", y_test)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (title, y_split) in zip(axes, splits):
        unique, counts = np.unique(y_split, return_counts=True)
        ax.bar(unique, counts)
        ax.set_title(title)
        ax.set_xlabel("Person ID")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=90)

    fig.suptitle("Class Distribution per Split (Stratified)")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved distribution plot to {save_path}")
    else:
        plt.show()


def save_splits(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    out_path="outputs/data_splits.npz",
):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        out_path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    print(f"Saved splits to {out_path}")


# Save scaler
def save_scaler(scaler, out_path="outputs/scaler.joblib"):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, out_path)
    print(f"Saved scaler to {out_path}")


# Save metadata
def save_metadata(metadata, out_path="outputs/metadata.json"):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {out_path}")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}.\n")

    images, labels = load_umist_mat(DATA_PATH)

    # Show/save sample images with labels (rubric)
    plot_sample_images(images, labels, n=12, save_path=os.path.join("outputs", "sample_images.png"))

    X = flatten_images(images)
    y = labels

    # Create DataFrame with labels (rubric) + save
    df = to_dataframe(X, y)
    df_out = os.path.join("outputs", "umist_dataframe.csv")
    os.makedirs(os.path.dirname(df_out), exist_ok=True)
    df.to_csv(df_out, index=False)
    print(f"Saved DataFrame to {df_out}")
    print(df.head())

    # Safety check for stratified split
    check_min_samples_per_class(y, min_required=3)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
    ) = split_and_scale(X, y)

    plot_path = os.path.join("outputs", "class_distribution_per_split.png")
    plot_class_distribution(y_train, y_val, y_test, save_path=plot_path)

    save_splits(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        out_path=os.path.join("outputs", "umist_splits.npz"),
    )

    # Save scaler for later steps
    save_scaler(scaler, out_path=os.path.join("outputs", "umist_scaler.joblib"))

    # Save metadata (helps report + reproducibility)
    metadata = {
        "random_state": RANDOM_STATE,
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        "shapes": {
            "images": list(images.shape),
            "X_flat": list(X.shape),
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "counts_total": pd.Series(y).value_counts().sort_index().to_dict(),
        "counts_train": pd.Series(y_train).value_counts().sort_index().to_dict(),
        "counts_val": pd.Series(y_val).value_counts().sort_index().to_dict(),
        "counts_test": pd.Series(y_test).value_counts().sort_index().to_dict(),
    }
    save_metadata(metadata, out_path=os.path.join("outputs", "metadata.json"))


if __name__ == "__main__":
    main()

