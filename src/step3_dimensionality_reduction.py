"""
STEP 3 – Dimensionality Reduction (COMP 257)

This script applies dimensionality reduction techniques to the UMIST dataset:
1. PCA (with multiple component sizes and explained variance analysis)
2. Autoencoder (non-linear latent representation)

Outputs:
- Explained variance plots (PCA)
- 2D PCA visualization
- Autoencoder latent space visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# For Autoencoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# ------------------------
# Paths
# ------------------------
OUT_DIR = "outputs"
SPLITS_PATH = os.path.join(OUT_DIR, "umist_splits.npz")
STEP3_DIR = os.path.join(OUT_DIR, "step3")
os.makedirs(STEP3_DIR, exist_ok=True)

RANDOM_STATE = 42


# ------------------------
# Load data
# ------------------------
def load_data(path):
    data = np.load(path)
    return data["X_train"], data["X_val"], data["X_test"], data["y_train"], data["y_val"], data["y_test"]


# ------------------------
# PCA SECTION
# ------------------------
def pca_explained_variance(X_train, components_list):
    explained = {}

    for n in components_list:
        pca = PCA(n_components=n, random_state=RANDOM_STATE)
        pca.fit(X_train)
        explained[n] = np.sum(pca.explained_variance_ratio_)

    return explained


def plot_explained_variance(explained_dict):
    components = list(explained_dict.keys())
    variance = list(explained_dict.values())

    plt.figure(figsize=(6, 4))
    plt.plot(components, variance, marker="o")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance vs Number of Components")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(STEP3_DIR, "pca_explained_variance.png"), dpi=150)
    plt.close()


def pca_2d_visualization(X, y):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab20", s=10)
    plt.colorbar(scatter, label="Person ID")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA 2D Projection of UMIST Faces")
    plt.tight_layout()
    plt.savefig(os.path.join(STEP3_DIR, "pca_2d_projection.png"), dpi=150)
    plt.close()

    return X_2d


# ------------------------
# AUTOENCODER SECTION
# ------------------------
def build_autoencoder(input_dim, latent_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(512, activation="relu")(input_layer)
    encoded = Dense(128, activation="relu")(encoded)
    latent = Dense(latent_dim, activation="linear", name="latent")(encoded)

    decoded = Dense(128, activation="relu")(latent)
    decoded = Dense(512, activation="relu")(decoded)
    output_layer = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=latent)

    autoencoder.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse"
    )

    return autoencoder, encoder


def train_autoencoder(autoencoder, X_train, X_val):
    history = autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=30,
        batch_size=64,
        shuffle=True,
        verbose=1
    )
    return history


def plot_autoencoder_latent(X_latent, y):
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_latent[:, 0], X_latent[:, 1], c=y, cmap="tab20", s=10)
    plt.colorbar(scatter, label="Person ID")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Autoencoder Latent Space (2D)")
    plt.tight_layout()
    plt.savefig(os.path.join(STEP3_DIR, "autoencoder_latent_2d.png"), dpi=150)
    plt.close()


# ------------------------
# MAIN
# ------------------------
def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(SPLITS_PATH)

    print("Loaded data:")
    print("X_train:", X_train.shape)

    # ----- PCA -----
    components_list = [10, 20, 50, 100]
    explained = pca_explained_variance(X_train, components_list)
    print("PCA explained variance:", explained)

    plot_explained_variance(explained)
    pca_2d_visualization(X_train, y_train)

    # ----- AUTOENCODER -----
    input_dim = X_train.shape[1]
    latent_dim = 2

    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
    train_autoencoder(autoencoder, X_train, X_val)

    X_latent = encoder.predict(X_train)
    plot_autoencoder_latent(X_latent, y_train)

    print("STEP 3 completed. Outputs saved to:", STEP3_DIR)


if __name__ == "__main__":
    main()

