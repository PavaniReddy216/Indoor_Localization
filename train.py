import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Flatten, Concatenate

# Load dataset
data = pd.read_csv(r"C:\MY PROJECTS\pro\TrainingData.csv")

# Handle missing values in RSSI columns (100 → -110 dBm)
X = data.iloc[:, 0:520].replace(100, -110)

# Standardize RSSI values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM (samples, timesteps, features)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Encode categorical labels
encoder_building = LabelEncoder()
encoder_floor = LabelEncoder()
encoder_room = LabelEncoder()

data["BUILDINGID"] = encoder_building.fit_transform(data["BUILDINGID"])
data["FLOOR"] = encoder_floor.fit_transform(data["FLOOR"])
data["SPACEID"] = encoder_room.fit_transform(data["SPACEID"])

# Define target labels
y_building = data["BUILDINGID"].values
y_floor = data["FLOOR"].values
y_room = data["SPACEID"].values

# Save encoders and scaler for later use
joblib.dump(encoder_building, "building_encoder.pkl")
joblib.dump(encoder_floor, "floor_encoder.pkl")
joblib.dump(encoder_room, "room_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# Split into training and testing data
X_train, X_test, y_train_building, y_test_building, y_train_floor, y_test_floor, y_train_room, y_test_room = train_test_split(
    X_lstm, y_building, y_floor, y_room, test_size=0.2, random_state=42
)

# ------------------ Hybrid ANN + LSTM Model ------------------ #

# Model Input
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# ----- LSTM Branch -----
lstm1 = LSTM(128, return_sequences=True)(input_layer)
dropout1 = Dropout(0.3)(lstm1)
lstm2 = LSTM(64, return_sequences=False)(dropout1)
lstm_bn = BatchNormalization()(lstm2)

# ----- ANN Branch -----
flat = Flatten()(input_layer)
dense1 = Dense(256, activation="relu")(flat)
dropout2 = Dropout(0.3)(dense1)
dense2 = Dense(128, activation="relu")(dropout2)
dense_bn = BatchNormalization()(dense2)

# ----- Fusion Layer (Combine LSTM & ANN) -----
merged = Concatenate()([lstm_bn, dense_bn])
final_dense = Dense(128, activation="relu")(merged)

# ----- Output Layers -----
building_output = Dense(len(encoder_building.classes_), activation="softmax", name="building_output")(final_dense)
floor_output = Dense(len(encoder_floor.classes_), activation="softmax", name="floor_output")(final_dense)
room_output = Dense(len(encoder_room.classes_), activation="softmax", name="room_output")(final_dense)

# Create Model
model = Model(inputs=input_layer, outputs=[building_output, floor_output, room_output])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
    loss={
        "building_output": "sparse_categorical_crossentropy",
        "floor_output": "sparse_categorical_crossentropy",
        "room_output": "sparse_categorical_crossentropy",
    },
    metrics={
        "building_output": ["accuracy"],
        "floor_output": ["accuracy"],
        "room_output": ["accuracy"],
    },
)

# Callbacks
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-5)
checkpoint = tf.keras.callbacks.ModelCheckpoint("indoor_localization_best_model.h5", monitor="val_loss", save_best_only=True, mode="min", verbose=1)

# Train Model
history = model.fit(
    X_train,
    {"building_output": y_train_building, "floor_output": y_train_floor, "room_output": y_train_room},
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[checkpoint, lr_scheduler],
)

# Save Final Model
model.save("indoor_localization_model.h5")

# Calculate Overall Accuracy
train_acc_building = history.history["building_output_accuracy"][-1]
train_acc_floor = history.history["floor_output_accuracy"][-1]
train_acc_room = history.history["room_output_accuracy"][-1]

overall_accuracy = (train_acc_building + train_acc_floor + train_acc_room) / 3
print(f"Overall Model Accuracy: {overall_accuracy * 100:.2f}%")
