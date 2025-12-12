"""
STEP 5 – Supervised Learning: Neural Network Classifier (COMP 257)

This script trains an ANN (MLP) to classify UMIST face images using
the preprocessed and scaled feature vectors.

Outputs:
- Training curves (loss & accuracy)
- Classification metrics on validation and test sets
- Sample test image predictions (true vs predicted labels)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------
# Paths
# ------------------------
OUT_DIR = "outputs"
SPLITS_PATH = os.path.join(OUT_DIR, "umist_splits.npz")
STEP5_DIR = os.path.join(OUT_DIR, "step5")
os.makedirs(STEP5_DIR, exist_ok=True)

RANDOM_STATE = 42
NUM_CLASSES = 20


# ------------------------
# Load data
# ------------------------
def load_data(path):
    data = np.load(path)
    return (
        data["X_train"],
        data["X_val"],
        data["X_test"],
        data["y_train"],
        data["y_val"],
        data["y_test"],
    )


# ------------------------
# Build ANN model
# ------------------------
def build_ann(input_dim, num_classes):
    model = Sequential([
        Dense(512, activation="relu", input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ------------------------
# Plot training curves
# ------------------------
def plot_training_history(history):
    # Loss
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(STEP5_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(STEP5_DIR, "accuracy_curve.png"), dpi=150)
    plt.close()


# ------------------------
# Plot sample predictions
# ------------------------
def plot_sample_predictions(X_test, y_true, y_pred, n=8):
    idx = np.random.choice(len(X_test), n, replace=False)

    # reshape flattened images back to original size (112x92)
    h, w = 112, 92
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for ax, i in zip(axes, idx):
        img = X_test[i].reshape(h, w)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {y_true[i]} | Pred: {y_pred[i]}")
        ax.axis("off")

    fig.suptitle("Sample Test Predictions")
    fig.tight_layout()
    plt.savefig(os.path.join(STEP5_DIR, "sample_predictions.png"), dpi=150)
    plt.close()


# ------------------------
# Main
# ------------------------
def main():
    print("=== STEP 5 STARTING ===")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(SPLITS_PATH)

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    model = build_ann(input_dim=X_train.shape[1], num_classes=NUM_CLASSES)
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    plot_training_history(history)

    # ------------------------
    # Evaluation
    # ------------------------
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_test_pred = np.argmax(model.predict(X_test), axis=1)

    print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    # Save predictions
    np.savez(
        os.path.join(STEP5_DIR, "predictions.npz"),
        y_test=y_test,
        y_test_pred=y_test_pred
    )

    plot_sample_predictions(X_test, y_test, y_test_pred)

    print("\nSTEP 5 COMPLETED.")
    print("Outputs saved to:", STEP5_DIR)


if __name__ == "__main__":
    main()

