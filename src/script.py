"""
Steps 1 & 2 COMP 257 project:
- Load UMIST dataset from umist_cropped.mat
- Explore structure
- Convert to feature matrix + labels
- Stratified train/validation/test split
- Normalize features
- Save split data for later stages
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.style.use("ggplot")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "umist_cropped.mat")
RANDOM_STATE = 42


def load_umist_mat(path: str) -> Tuple[np.ndarray, np.ndarray]:

    mat = sio.loadmat(path)

    if "facedat" not in mat:
        raise KeyError(" No File Found")

    facedat = mat["facedat"]
    # Normalize facedat into a 1D sequence of subject arrays\
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
            raise ValueError(
                f"Wrong shape for subject {person_idx}: {subject_arr.shape}. "
            )

        h, w, num_images = subject_arr.shape

        # Loop through each image for this subject and append to the lists.
        for j in range(num_images):
            img = subject_arr[:, :, j]
            images_list.append(img.astype(np.float32))
            labels_list.append(person_idx)

    images = np.stack(images_list, axis=0)
    labels = np.array(labels_list, dtype=np.int64)

    print(f"Images shape: {images.shape} ")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of subjects: {len(subjects)}")

    return images, labels


# Flatten images into vectors of shape
def flatten_images(images: np.ndarray) -> np.ndarray:

    if images.ndim != 3:
        raise ValueError(f"Image Shape iS incorrect")

    n_samples, h, w = images.shape
    X = images.reshape(n_samples, h * w)
    print(f"Flattened images to shape: {X.shape} ")
    return X


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


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}.\n")

    images, labels = load_umist_mat(DATA_PATH)

    X = flatten_images(images)
    y = labels

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


if __name__ == "__main__":
    main()
