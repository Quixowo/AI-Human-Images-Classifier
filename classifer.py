
"""
# AI vs. Real Image Classifier

Fine-tunes MobileNetV3-Large (ImageNet weights) to tell AI-generated images from real photos.

**Data split (stratified, fixed seed):** 70% train / 15% validation / 15% test.
- **Train:** augmented with horizontal flips and width/height shifts
- **Validation:** no augmentation; used for learning-rate scheduling, early stopping, and picking the best checkpoint
- **Test:** no augmentation; never touched during training, evaluated exactly once at the end

Runtime: Colab GPU (T4). Go to Runtime > Change runtime type > T4 GPU before running.
"""

# --- Environment Setup & Imports ---
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ReduceLROnPlateau, ModelCheckpoint, EarlyStopping,
                                        BackupAndRestore, CSVLogger)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

# Reproducibility: fixes the data split and weight init (GPU training can still vary slightly run to run)
SEED = 42
tf.keras.utils.set_random_seed(SEED)

print("TensorFlow Version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# --- Save everything to Google Drive so progress survives a Colab disconnect ---
from google.colab import drive
drive.mount("/content/drive")
SAVE_DIR = "/content/drive/MyDrive/ai-image-classifier"
os.makedirs(SAVE_DIR, exist_ok=True)
print("Saving checkpoints and results to:", SAVE_DIR)

# --- Data Acquisition ---
# Download dataset (only runs if not already present)
path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")
print("Dataset base path:", path)

# ---- Data Preparation ----

# Load train CSV
train_csv = os.path.join(path, "train.csv")
train_df = pd.read_csv(train_csv)

# Create filtered DataFrames with full paths

real_df = train_df[train_df['label'] == 0].reset_index(drop=True)
fake_df = train_df[train_df['label'] == 1].reset_index(drop=True)

real_df['full_path'] = real_df['file_name'].apply(lambda x: os.path.join(path, x))
fake_df['full_path'] = fake_df['file_name'].apply(lambda x: os.path.join(path, x))
real_df = real_df.drop(columns = ['Unnamed: 0', 'file_name'])
fake_df = fake_df.drop(columns = ['Unnamed: 0', 'file_name'])

# --- 1. Combine and Split (70% train / 15% validation / 15% test) ---
all_data_df = pd.concat([real_df, fake_df]).reset_index(drop=True)
all_data_df["label"] = all_data_df["label"].astype(str)   # generators expect string class names
print(f"Total combined images: {len(all_data_df)}")

# Stratified splits keep the real/AI ratio identical in every subset
train_df, temp_df = train_test_split(
    all_data_df, test_size=0.30, stratify=all_data_df["label"], random_state=SEED)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)

# Sanity checks: no image appears in more than one split
assert set(train_df.full_path).isdisjoint(val_df.full_path)
assert set(train_df.full_path).isdisjoint(test_df.full_path)
assert set(val_df.full_path).isdisjoint(test_df.full_path)

for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    ai_share = (df["label"] == "1").mean()
    print(f"{name:<11} {len(df):>6} images  ({ai_share:.1%} AI-generated)")

# --- 2. Configuration ---
TARGET_IMG_SIZE = (256, 256)
BATCH_SIZE = 32

# --- 3. Data Generators ---
# Augmentation is applied ONLY to training data.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode="nearest",
)
# Validation and test images are only preprocessed, never augmented,
# so evaluation reflects the images as they really are.
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

common = dict(x_col="full_path", y_col="label", target_size=TARGET_IMG_SIZE,
              batch_size=BATCH_SIZE, class_mode="binary")

print("Creating Training Generator...")
train_generator = train_datagen.flow_from_dataframe(train_df, shuffle=True, seed=SEED, **common)
print("Creating Validation Generator...")
validation_generator = eval_datagen.flow_from_dataframe(val_df, shuffle=False, **common)
print("Creating Test Generator...")
test_generator = eval_datagen.flow_from_dataframe(test_df, shuffle=False, **common)

# Label mapping used by the demo app: 0 = real, 1 = AI-generated
assert train_generator.class_indices == {"0": 0, "1": 1}, train_generator.class_indices
print("Class indices:", train_generator.class_indices)

MODEL_PATH     = os.path.join(SAVE_DIR, "ai_vs_human_mobilenetv3.keras")  # best checkpoint
PHASE1_WEIGHTS = os.path.join(SAVE_DIR, "phase1.weights.h5")
PHASE1_LOG     = os.path.join(SAVE_DIR, "phase1_log.csv")
PHASE2_LOG     = os.path.join(SAVE_DIR, "phase2_log.csv")
BACKUP_DIR     = os.path.join(SAVE_DIR, "backup")          # mid-training resume state
DONE_FLAG      = os.path.join(SAVE_DIR, "TRAINING_DONE")

# --- 4. Build Optimized MobileNetV3Large ---
# All training artifacts live on Google Drive. If Colab disconnects, just rerun the
# notebook from the top: training resumes from the last finished epoch.

base_model = MobileNetV3Large(
    input_shape=(TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)

# Freeze all layers initially
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

METRICS = ["accuracy", tf.keras.metrics.Precision(name="precision"),
           tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")]

if os.path.exists(DONE_FLAG):
    print("Training already finished. Best model is at", MODEL_PATH)
    print("Delete the folder", SAVE_DIR, "if you want to retrain from scratch.")
else:
    # --- 5. Training Phase 1: Warm-up with frozen base (1 epoch) ---
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=METRICS)
    if os.path.exists(PHASE1_WEIGHTS):
        print("Phase 1 already done, loading its weights from Drive.")
        model.load_weights(PHASE1_WEIGHTS)
    else:
        print("\nTraining phase 1: Transfer Learning (Base Frozen)")
        model.fit(train_generator, validation_data=validation_generator, epochs=1,
                  callbacks=[CSVLogger(PHASE1_LOG)])
        model.save_weights(PHASE1_WEIGHTS)

    # --- 6. Training Phase 2: Fine-tuning top 50 layers (up to 9 epochs) ---
    # BatchNormalization layers stay frozen: unfreezing them lets small batches overwrite
    # the ImageNet statistics, which destabilizes fine-tuning (standard Keras guidance)
    for layer in base_model.layers[-50:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    print("Trainable layers in base:", sum(l.trainable for l in base_model.layers))

    # On a resumed run, remember the best validation loss so far so an
    # older, better checkpoint never gets overwritten by a worse one
    best_so_far = None
    if os.path.exists(PHASE2_LOG) and os.path.exists(MODEL_PATH):
        best_so_far = float(pd.read_csv(PHASE2_LOG)["val_loss"].min())
        print(f"Resuming phase 2. Best validation loss so far: {best_so_far:.4f}")

    callbacks = [
        # If validation loss plateaus, drop LR to squeeze out accuracy
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
        # Keep only the checkpoint with the best validation loss
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True,
                        initial_value_threshold=best_so_far, verbose=1),
        # Stop if validation loss hasn't improved in 3 epochs
        EarlyStopping(monitor="val_loss", patience=3, verbose=1),
        # Per-epoch metrics on Drive (used for the training curves, survives disconnects)
        CSVLogger(PHASE2_LOG, append=True),
        # Saves full training state (weights, optimizer, learning rate, epoch) so a rerun
        # resumes after a disconnect. Listed last so it captures this epoch's LR change.
        BackupAndRestore(backup_dir=BACKUP_DIR),
    ]

    model.compile(optimizer=Adam(learning_rate=5e-5), loss="binary_crossentropy", metrics=METRICS)

    print("\nTraining phase 2: Fine-tuning")
    model.fit(train_generator, validation_data=validation_generator,
              epochs=9, callbacks=callbacks)

    open(DONE_FLAG, "w").close()
    print("\nTraining Complete. Best checkpoint saved to", MODEL_PATH)

# --- 7. Training Curves (read from the logs on Drive, so they include resumed runs) ---
hist = pd.concat([pd.read_csv(PHASE1_LOG), pd.read_csv(PHASE2_LOG)], ignore_index=True)
epochs = range(1, len(hist) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ["loss", "accuracy"]):
    ax.plot(epochs, hist[metric], marker="o", label="Train")
    ax.plot(epochs, hist[f"val_{metric}"], marker="o", label="Validation")
    ax.axvline(1.5, color="gray", linestyle="--", linewidth=1)   # phase 1 -> phase 2
    ax.set_title(metric.capitalize()); ax.set_xlabel("Epoch"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
plt.show()

"""## Final evaluation on the held-out test set

Run this **once**, after training is finished. The test set was never used for training, learning-rate scheduling, early stopping, or checkpoint selection, so these numbers are an honest estimate of performance on new images.
"""

# --- 8. Test Set Evaluation ---
best_model = load_model(MODEL_PATH)

print("Predicting on test set...")
y_prob = best_model.predict(test_generator, verbose=1).ravel()
y_true = np.array(test_generator.classes)          # same order as predictions (shuffle=False)
y_pred = (y_prob >= 0.5).astype(int)

test_metrics = {
    "test_images": int(len(y_true)),
    "accuracy":  float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    "roc_auc":   float(roc_auc_score(y_true, y_prob)),
    "loss":      float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))),
}

print("\n===== Held-out Test Results =====")
print(f"{'test_images':<12} {test_metrics['test_images']}")
for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    print(f"{k:<12} {test_metrics[k]:.4%}")
print(f"{'loss':<12} {test_metrics['loss']:.4f}")

print("\n" + classification_report(y_true, y_pred, target_names=["Real", "AI-generated"], digits=4, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
test_metrics["confusion_matrix"] = cm.tolist()
ConfusionMatrixDisplay(cm, display_labels=["Real", "AI-generated"]).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Held-out Test Set)")
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.show()

with open(os.path.join(SAVE_DIR, "test_metrics.json"), "w") as f:
    json.dump(test_metrics, f, indent=2)

# --- 9. Download the model and results ---
from google.colab import files
# Everything is also saved in your Google Drive under MyDrive/ai-image-classifier
for fname in ["ai_vs_human_mobilenetv3.keras", "test_metrics.json", "confusion_matrix.png", "training_curves.png"]:
    files.download(os.path.join(SAVE_DIR, fname))

"""## Export for the in-browser demo (ONNX)

Converts the best model to ONNX so it can run in a web browser, checks the converted model matches the original, and downloads `model.onnx`.
"""

# --- 10. Convert to ONNX for the web demo ---
!pip install -q tf2onnx onnxruntime

import tf2onnx
import onnxruntime as ort

ONNX_PATH = os.path.join(SAVE_DIR, "model.onnx")
spec = (tf.TensorSpec((None, TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1], 3), tf.float32, name="image"),)
tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=17, output_path=ONNX_PATH)

# Verify ONNX predictions match Keras on real test images
x_batch, _ = test_generator[0]
keras_preds = best_model(x_batch, training=False).numpy().ravel()
onnx_preds = ort.InferenceSession(ONNX_PATH).run(None, {"image": x_batch.astype("float32")})[0].ravel()
max_diff = float(np.abs(keras_preds - onnx_preds).max())
print(f"Max Keras vs ONNX difference: {max_diff:.2e}", "(OK)" if max_diff < 1e-3 else "(MISMATCH, do not deploy)")

files.download(ONNX_PATH)

