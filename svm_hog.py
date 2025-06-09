import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import pandas as pd
import joblib
import time
import json
from utils import load_dataset, plot_confusion_matrix
import matplotlib.pyplot as plt

def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=False):
    """
    Extract HOG features from images
    
    Parameters:
    - images: Array of images
    - orientations: Number of orientation bins (default: 9)
    - pixels_per_cell: Size of cell in pixels (default: 8x8)
    - cells_per_block: Number of cells per block (default: 2x2)
    """
    features = []
    for image in images:
        # Extract HOG features
        hog_features = hog(image, 
                         orientations=orientations,
                         pixels_per_cell=pixels_per_cell,
                         cells_per_block=cells_per_block,
                         block_norm='L2-Hys',
                         visualize=visualize,
                         feature_vector=True)
        features.append(hog_features)
    
    return np.array(features)

def train_svm_model(data_dir, model_save_path):
    """
    Train SVM model with HOG features for emotion recognition
    """
    # Track training time
    start_time = time.time()
    
    # Load dataset
    print("Loading training dataset...")
    X_train, y_train = load_dataset(data_dir, mode='train')
    
    print("Loading test dataset...")
    X_test, y_test = load_dataset(data_dir, mode='test')
    
    # Convert one-hot to labels (y is already one-hot from load_dataset)
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Split training data for validation
    X_train, X_val, y_train_labels, y_val_labels = train_test_split(
        X_train, y_train_labels, test_size=0.2, random_state=42, stratify=y_train_labels
    )
    
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Extract HOG features
    print("\nExtracting HOG features...")
    
    # Test different HOG parameters
    hog_params = {
        'orientations': 9,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2)
    }
    
    X_train_hog = extract_hog_features(X_train, **hog_params)
    X_val_hog = extract_hog_features(X_val, **hog_params)
    X_test_hog = extract_hog_features(X_test, **hog_params)
    
    print(f"HOG feature vector size: {X_train_hog.shape[1]}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_hog = scaler.fit_transform(X_train_hog)
    X_val_hog = scaler.transform(X_val_hog)
    X_test_hog = scaler.transform(X_test_hog)
    
    # Extensive Grid Search for SVM hyperparameters
    print("\nPerforming Grid Search for optimal hyperparameters...")
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'degree': [2, 3, 4]  # Only for poly kernel
    }
    
    svm_model = svm.SVC(probability=True, random_state=42)
    
    # Grid search với cross-validation
    grid_search = GridSearchCV(
        svm_model, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train_hog, y_train_labels)
    
    # Best parameters
    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Use best model
    best_svm = grid_search.best_estimator_
    
    # Validation accuracy
    val_predictions = best_svm.predict(X_val_hog)
    val_accuracy = accuracy_score(y_val_labels, val_predictions)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    test_predictions = best_svm.predict(X_test_hog)
    test_accuracy = accuracy_score(y_test_labels, test_predictions)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Get prediction probabilities
    test_proba = best_svm.predict_proba(X_test_hog)
    test_loss = -np.mean(np.log(test_proba[np.arange(len(y_test_labels)), y_test_labels] + 1e-7))
    print(f"Test loss: {test_loss:.4f}")
    
    # Emotion labels
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_labels, test_predictions)
    cm_df = pd.DataFrame(cm, index=emotions, columns=emotions)
    print("\nConfusion Matrix:")
    print(cm_df)
    cm_df.to_csv("models/confusion_matrix_svm.csv")
    
    # Classification Report
    report = classification_report(y_test_labels, test_predictions, 
                                 target_names=emotions, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Save detailed report
    report_dict = classification_report(y_test_labels, test_predictions, 
                                      target_names=emotions, output_dict=True)
    
    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i, emotion in enumerate(emotions):
        mask = y_test_labels == i
        if np.sum(mask) > 0:
            per_class_accuracy[emotion] = np.mean(test_predictions[mask] == i)
    
    # Cross-validation scores
    print("\nCross-validation analysis...")
    cv_scores = cross_val_score(best_svm, X_train_hog, y_train_labels, cv=5)
    print(f"CV scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save comprehensive results
    results = {
        'model_type': 'SVM+HOG',
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'validation_accuracy': float(val_accuracy),
        'training_time_seconds': training_time,
        'hog_parameters': hog_params,
        'hog_feature_size': int(X_train_hog.shape[1]),
        'best_svm_parameters': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'per_class_accuracy': per_class_accuracy,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'dataset_info': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
    }
    
    with open('models/svm_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    with open('models/classification_report_svm.txt', 'w') as f:
        f.write(f"SVM+HOG Model - Classification Report\n")
        f.write(f"Training Time: {training_time/60:.2f} minutes\n")
        f.write(f"HOG Feature Size: {X_train_hog.shape[1]}\n")
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write("="*50 + "\n\n")
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, emotions, 'SVM+HOG')
    
    # Visualize HOG features for a sample image
    visualize_hog_features(X_test[0], hog_params)
    
    # Save model and scaler
    joblib.dump(best_svm, model_save_path)
    joblib.dump(scaler, 'models/scaler_svm.pkl')
    print(f"\nModel saved to {model_save_path}")
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    return best_svm, results

def visualize_hog_features(image, hog_params):
    """
    Visualize HOG features for a sample image
    """
    hog_features, hog_image = hog(image, 
                                  orientations=hog_params['orientations'],
                                  pixels_per_cell=hog_params['pixels_per_cell'],
                                  cells_per_block=hog_params['cells_per_block'],
                                  block_norm='L2-Hys',
                                  visualize=True,
                                  feature_vector=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('HOG Features Visualization')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/hog_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("HOG visualization saved to plots/hog_visualization.png")

if __name__ == "__main__":
    # Define paths
    DATA_DIR = "data"
    MODEL_SAVE_PATH = "models/emotion_model_svm.pkl"
    
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Train model
    model, results = train_svm_model(DATA_DIR, MODEL_SAVE_PATH) 