import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_emotion_model, save_model
from utils import load_dataset, plot_training_history, plot_confusion_matrix, save_training_report
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import time
import json

def train_model(data_dir, model_save_path, batch_size=8, epochs=50):  # Giảm batch size và epochs
    """
    Train the emotion recognition model with optimized parameters for CPU
    """
    # Track training time
    start_time = time.time()
    
    # Load and preprocess dataset
    print("Loading training dataset...")
    X_train, y_train = load_dataset(data_dir, mode='train')
    
    print("Loading test dataset...")
    X_test, y_test = load_dataset(data_dir, mode='test')
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Reshape data for CNN input
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    print("\nCreating model...")
    model = create_emotion_model()
    
    # Count parameters
    trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
    non_trainable_params = sum([np.prod(w.shape) for w in model.non_trainable_weights])
    print(f"Total parameters: {trainable_params + non_trainable_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    # Simplified Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,        # Giảm từ 15 xuống 10
        width_shift_range=0.1,    # Giảm từ 0.15 xuống 0.1
        height_shift_range=0.1,   # Giảm từ 0.15 xuống 0.1
        horizontal_flip=True,
        zoom_range=0.1,          # Giảm từ 0.15 xuống 0.1
        fill_mode='nearest'
    )
    
    # Chỉ augment training data, không augment validation
    datagen.fit(X_train)
    
    # Define callbacks with adjusted parameters
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=30,          # Tăng từ 20 lên 30
            verbose=1,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,          # Tăng từ 10 lên 15
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger('models/training_log.csv')
    ]
    
    # Train model với augmented data
    print("\nTraining model with optimized parameters...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Predictions for detailed analysis
    y_pred = model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    
    # Emotion labels
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_df = pd.DataFrame(cm, index=emotions, columns=emotions)
    print("\nConfusion Matrix:")
    print(cm_df)
    cm_df.to_csv("models/confusion_matrix_cnn.csv")
    
    # Classification Report
    report = classification_report(y_true_labels, y_pred_labels, target_names=emotions, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Save detailed report
    report_dict = classification_report(y_true_labels, y_pred_labels, target_names=emotions, output_dict=True)
    
    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i, emotion in enumerate(emotions):
        mask = y_true_labels == i
        if np.sum(mask) > 0:
            per_class_accuracy[emotion] = np.mean(y_pred_labels[mask] == i)
    
    # Save comprehensive results
    results = {
        'model_type': 'CNN',
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'training_time_seconds': training_time,
        'total_parameters': int(trainable_params + non_trainable_params),
        'trainable_parameters': int(trainable_params),
        'epochs_trained': len(history.history['loss']),
        'batch_size': batch_size,
        'learning_rate': 0.0001,
        'per_class_accuracy': per_class_accuracy,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'dataset_info': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
    }
    
    with open('models/cnn_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    with open('models/classification_report_cnn.txt', 'w') as f:
        f.write(f"CNN Model - Classification Report\n")
        f.write(f"Training Time: {training_time/60:.2f} minutes\n")
        f.write(f"Total Parameters: {trainable_params:,}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write("="*50 + "\n\n")
        f.write(report)
    
    # Plot training history
    plot_training_history(history, model_name='CNN')
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, emotions, 'CNN')
    
    # Save final model
    save_model(model, model_save_path)
    
    # Generate training report
    save_training_report(history, results, 'CNN')
    
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    
    return model, history, results

if __name__ == "__main__":
    # Define paths
    DATA_DIR = "data"
    MODEL_SAVE_PATH = "models/emotion_model_cnn.h5"
    
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Train model
    model, history, results = train_model(DATA_DIR, MODEL_SAVE_PATH) 