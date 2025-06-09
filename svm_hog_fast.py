import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import pandas as pd
import joblib
import time
import json
from utils import load_dataset, plot_confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=False):
    """
    Extract HOG features from images
    """
    features = []
    for image in images:
        hog_features = hog(image, 
                         orientations=orientations,
                         pixels_per_cell=pixels_per_cell,
                         cells_per_block=cells_per_block,
                         block_norm='L2-Hys',
                         visualize=visualize,
                         feature_vector=True)
        features.append(hog_features)
    
    return np.array(features)

def train_svm_model_fast(data_dir, model_save_path, use_subset=True, subset_ratio=0.3):
    """
    Train SVM model with optimized settings for faster training
    
    Parameters:
    - use_subset: If True, use only a subset of data for faster training
    - subset_ratio: Ratio of data to use (0.3 = 30% of data)
    """
    start_time = time.time()
    
    # Load dataset
    print("Loading training dataset...")
    X_train, y_train = load_dataset(data_dir, mode='train')
    
    print("Loading test dataset...")
    X_test, y_test = load_dataset(data_dir, mode='test')
    
    # Convert one-hot to labels
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Option to use subset for faster training
    if use_subset:
        print(f"\n⚡ Using {subset_ratio*100:.0f}% subset for faster training...")
        subset_size = int(len(X_train) * subset_ratio)
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train = X_train[indices]
        y_train_labels = y_train_labels[indices]
    
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
    
    # OPTIMIZED APPROACH
    print("\n" + "="*60)
    print("🚀 USING OPTIMIZED TRAINING STRATEGY")
    print("="*60)
    
    # Step 1: Quick test with linear SVM (fastest)
    print("\n1️⃣ Testing Linear SVM (fastest)...")
    linear_svm = svm.SVC(kernel='linear', C=1.0, random_state=42)
    linear_start = time.time()
    linear_svm.fit(X_train_hog, y_train_labels)
    linear_time = time.time() - linear_start
    linear_acc = accuracy_score(y_val_labels, linear_svm.predict(X_val_hog))
    print(f"   Linear SVM: {linear_acc:.4f} accuracy in {linear_time:.1f}s")
    
    # Step 2: Quick test with RBF SVM
    print("\n2️⃣ Testing RBF SVM with default params...")
    rbf_svm = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    rbf_start = time.time()
    rbf_svm.fit(X_train_hog, y_train_labels)
    rbf_time = time.time() - rbf_start
    rbf_acc = accuracy_score(y_val_labels, rbf_svm.predict(X_val_hog))
    print(f"   RBF SVM: {rbf_acc:.4f} accuracy in {rbf_time:.1f}s")
    
    # Step 3: Fine-tune the better kernel
    print("\n3️⃣ Fine-tuning best kernel...")
    
    if linear_acc > rbf_acc:
        print("   → Linear kernel performed better, fine-tuning C parameter...")
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100]
        }
        base_model = svm.SVC(kernel='linear', probability=True, random_state=42)
    else:
        print("   → RBF kernel performed better, fine-tuning C and gamma...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 0.001, 0.01, 0.1]
        }
        base_model = svm.SVC(kernel='rbf', probability=True, random_state=42)
    
    # Use only 3-fold CV for faster training
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=3,  # Reduced from 5 to 3
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\n   Grid Search with {len(param_grid['C']) * len(param_grid.get('gamma', [1]))} candidates...")
    grid_start = time.time()
    grid_search.fit(X_train_hog, y_train_labels)
    grid_time = time.time() - grid_start
    
    print(f"\n   ✅ Best parameters: {grid_search.best_params_}")
    print(f"   ✅ Best CV accuracy: {grid_search.best_score_:.4f}")
    print(f"   ⏱️  Grid search time: {grid_time/60:.1f} minutes")
    
    # Use best model
    best_svm = grid_search.best_estimator_
    
    # Validation accuracy
    val_predictions = best_svm.predict(X_val_hog)
    val_accuracy = accuracy_score(y_val_labels, val_predictions)
    print(f"\n   Validation accuracy: {val_accuracy:.4f}")
    
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
    cm_df.to_csv("models/confusion_matrix_svm_fast.csv")
    
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
    
    # Save comprehensive results
    results = {
        'model_type': 'SVM+HOG (Fast)',
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'validation_accuracy': float(val_accuracy),
        'training_time_seconds': training_time,
        'used_subset': use_subset,
        'subset_ratio': subset_ratio if use_subset else 1.0,
        'hog_parameters': hog_params,
        'hog_feature_size': int(X_train_hog.shape[1]),
        'best_svm_parameters': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'per_class_accuracy': per_class_accuracy,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'dataset_info': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
    }
    
    with open('models/svm_results_fast.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save model and scaler
    joblib.dump(best_svm, model_save_path.replace('.pkl', '_fast.pkl'))
    joblib.dump(scaler, 'models/scaler_svm_fast.pkl')
    
    print(f"\n✨ Training completed in {training_time/60:.1f} minutes!")
    print(f"   (Original approach would take ~{187.5*24:.0f} hours)")
    
    return best_svm, results

def train_with_randomized_search(data_dir, model_save_path):
    """
    Alternative: Use RandomizedSearchCV for even faster exploration of larger parameter space
    """
    start_time = time.time()
    
    # Load dataset
    print("Loading dataset...")
    X_train, y_train = load_dataset(data_dir, mode='train')
    X_test, y_test = load_dataset(data_dir, mode='test')
    
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Use 30% subset
    subset_size = int(len(X_train) * 0.3)
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_train = X_train[indices]
    y_train_labels = y_train_labels[indices]
    
    # Split and extract features
    X_train, X_val, y_train_labels, y_val_labels = train_test_split(
        X_train, y_train_labels, test_size=0.2, random_state=42
    )
    
    print("Extracting HOG features...")
    X_train_hog = extract_hog_features(X_train)
    X_val_hog = extract_hog_features(X_val)
    X_test_hog = extract_hog_features(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_hog = scaler.fit_transform(X_train_hog)
    X_val_hog = scaler.transform(X_val_hog)
    X_test_hog = scaler.transform(X_test_hog)
    
    # Randomized search with more parameters
    param_dist = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4, 5]
    }
    
    print("\n🎲 Using RandomizedSearchCV (testing 30 random combinations)...")
    
    random_search = RandomizedSearchCV(
        svm.SVC(probability=True, random_state=42),
        param_distributions=param_dist,
        n_iter=30,  # Test only 30 random combinations
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train_hog, y_train_labels)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV accuracy: {random_search.best_score_:.4f}")
    
    # Evaluate
    test_acc = accuracy_score(y_test_labels, random_search.predict(X_test_hog))
    print(f"Test accuracy: {test_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"\n✨ Total time: {total_time/60:.1f} minutes")
    
    return random_search.best_estimator_

if __name__ == "__main__":
    # Define paths
    DATA_DIR = "data"
    MODEL_SAVE_PATH = "models/emotion_model_svm.pkl"
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print("Choose training approach:")
    print("1. Fast Grid Search (recommended) - ~30 minutes")
    print("2. Randomized Search - ~20 minutes")
    print("3. Full dataset with optimized grid - ~2-3 hours")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Fast approach with subset
        model, results = train_svm_model_fast(DATA_DIR, MODEL_SAVE_PATH, 
                                            use_subset=True, subset_ratio=0.3)
    elif choice == '2':
        # Randomized search
        model = train_with_randomized_search(DATA_DIR, MODEL_SAVE_PATH)
    else:
        # Full dataset but optimized
        model, results = train_svm_model_fast(DATA_DIR, MODEL_SAVE_PATH, 
                                            use_subset=False) 