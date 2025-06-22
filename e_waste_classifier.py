# e_waste_classifier.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ================================
# 1. Configuration
# ================================
DATA_DIRS = {
    'train': r'modified-dataset/train',
    'val': r'modified-dataset/val',
    'test': r'modified-dataset/test'
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = 'e_waste_classifier.h5'


# ================================
# 2. Data Preparation
# ================================
def load_data(train_dir, val_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_data, val_data


# ================================
# 3. Build Model
# ================================
def build_model(num_classes):
    base_model = EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ================================
# 4. Train Model
# ================================
def train_model(model, train_data, val_data):
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

    history = model.fit(
        train_data,
        epochs=20,
        validation_data=val_data,
        callbacks=[early_stop, reduce_lr]
    )
    return model, history


# ================================
# 5. Save or Load Model
# ================================
def save_or_load_model(model_path, model=None):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return load_model(model_path)
    else:
        print(f"Saving model to {model_path}")
        model.save(model_path)
        return model


# ================================
# 6. Evaluate Model
# ================================
def evaluate_model(model, val_data):
    val_data.reset()
    predictions = model.predict(val_data, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_data.classes
    class_labels = list(val_data.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return class_labels


# ================================
# 7. Plot Training History
# ================================
def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ================================
# 8. Show Sample Predictions
# ================================
def show_sample_predictions(model, val_data, class_labels, num_samples=4):
    val_data.reset()
    images, labels = next(val_data)
    preds = model.predict(images)
    predicted_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels, axis=1)

    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(f"True: {class_labels[true_classes[i]]}\nPred: {class_labels[predicted_classes[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# ================================
# 9. Main Pipeline
# ================================
def main():
    print("🚀 Loading data...")
    train_data, val_data = load_data(DATA_DIRS['train'], DATA_DIRS['val'], IMG_SIZE, BATCH_SIZE)

    if os.path.exists(MODEL_PATH):
        print("📦 Loading existing model...")
        model = load_model(MODEL_PATH)
        history = None  # No history available if not training
    else:
        print("🧠 Building and training model...")
        model = build_model(train_data.num_classes)
        model, history = train_model(model, train_data, val_data)
        model.save(MODEL_PATH)

    if history:
        print("📊 Plotting training history...")
        plot_training_history(history)

    print("📈 Evaluating model...")
    class_labels = evaluate_model(model, val_data)

    print("🖼 Showing predictions...")
    show_sample_predictions(model, val_data, class_labels)



if __name__ == "__main__":
    main()