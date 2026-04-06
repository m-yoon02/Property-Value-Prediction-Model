# CPSC 383 (Fall 2025) — Assignment 3
# Michelle Yoon (30189382)
# Final Model Code (Model 3: Best MAE)
#
# This script loads the property value dataset for the City of Calgary,
# performs full preprocessing (feature cleaning, encoding, normalization),
# constructs a neural network using TensorFlow/Keras, trains the model,
# evaluates its performance, and prints the final Mean Absolute Error (MAE).
#
# This is the FINAL model (Model Version 3) that achieves the lowest MAE.
# Assignment instructions specify that ONLY this final model code is submitted
# as main.py, while underfitting and overfitting models appear only in the report.
#
# Allowed libraries: numpy, pandas, tensorflow, keras, matplotlib (optional)
# No additional external libraries are used.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
# Load the full 600k property dataset. The CSV must be in the same directory.
DATA_FILE = "yyc_pv.csv"
df = pd.read_csv(DATA_FILE)

# -------------------------------------------------------------
# 2. FEATURE ENGINEERING & ENCODING
# -------------------------------------------------------------
# The original dataset contains multiple types of features:
# - Categorical (COMM_CODE, ASSESSMENT_CLASS, PROPERTY_TYPE)
# - Numerical (LAT, LON, LAND_SIZE_SM, LAND_SIZE_SF, YEAR_OF_CONSTRUCTION)
# - Text (ADDRESS, COMM_NAME, ASSESSMENT_CLASS_DESCRIPTION)
#
# Text-based columns do not provide structured numerical information and
# greatly increase dimensionality without improving model performance,
# so we remove them.

df = df.drop(columns=["ADDRESS", "COMM_NAME", "ASSESSMENT_CLASS_DESCRIPTION"])

# One-hot encode categorical features so the neural network can interpret categories.
categorical_cols = ["COMM_CODE", "ASSESSMENT_CLASS", "PROPERTY_TYPE"]
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Normalize all numerical features to improve model convergence.
# Normalization ensures that large-scale features (e.g., land size)
# do not dominate training relative to small-scale features (e.g., year built).
numeric_cols = ["YEAR_OF_CONSTRUCTION", "LAT", "LON", "LAND_SIZE_SM", "LAND_SIZE_SF"]
for col in numeric_cols:
    mean = df_encoded[col].mean()
    std = df_encoded[col].std()
    df_encoded[col] = (df_encoded[col] - mean) / std

# Separate input features (X) from target labels (y = assessed property value)
labels = df_encoded["ASSESSED_VALUE"].values.astype(np.float32)
features = df_encoded.drop(columns=["ASSESSED_VALUE"]).values.astype(np.float32)

print("Total feature vector size:", features.shape[1])

# -------------------------------------------------------------
# 3. SHUFFLE + TRAIN/TEST SPLIT (5:1 ratio)
# -------------------------------------------------------------
# Shuffle the dataset manually by zipping features & labels together.
# This ensures random distribution since the original dataset had no strict order.
combined = list(zip(features, labels))
random.shuffle(combined)

# Unzip the shuffled dataset back into features and labels
features, labels = zip(*combined)
features = np.array(features, dtype=np.float32)
labels = np.array(labels, dtype=np.float32)

# Create a 5:1 train/test split (required by assignment instructions)
cut = int(len(features) * (5/6))
X_train, X_test = features[:cut], features[cut:]
y_train, y_test = labels[:cut], labels[cut:]

print("Training size:", len(X_train))
print("Testing size:", len(X_test))

# -------------------------------------------------------------
# 4. MODEL 3 — FINAL MODEL (lowest MAE)
# -------------------------------------------------------------
# This model is designed to balance complexity and generalization.
# It uses:
# - Multiple dense layers (256 → 256 → 128 → 64 neurons)
# - ReLU activation
# - L2 regularization to prevent overfitting
# - Adam optimizer with reduced learning rate (0.0005)
# - MAE loss function (assignment requirement)
#
# This architecture consistently achieves MAE under ~300k.

model = keras.Sequential([
    # Input layer: matches the dimension of the feature vector
    layers.Input(shape=(features.shape[1],)),

    # First hidden layer (large size, with L2 regularization)
    layers.Dense(256, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.0001)),

    # Second hidden layer (same size + L2 regularization)
    layers.Dense(256, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.0001)),

    # Third and fourth layers (reduced size to encourage compression)
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),

    # Output layer for regression (predicts a single numeric value)
    layers.Dense(1)
])

# Compile the model with Adam optimizer and MAE loss.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='mae',
    metrics=['mae']
)

model.summary()  # Print model structure for verification

# -------------------------------------------------------------
# 5. TRAIN MODEL
# -------------------------------------------------------------
# Train the model using 10% of the training set as validation data.
# Epoch count (25) provides enough time to learn without overfitting.
EPOCHS = 25
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=64,
    verbose=1  # Print training progress
)

# -------------------------------------------------------------
# 6. TEST MODEL
# -------------------------------------------------------------
# Evaluate the model’s performance on the test set.
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)

print("\n======================================")
print("FINAL MODEL — TEST MAE:", test_mae)
print("======================================\n")


