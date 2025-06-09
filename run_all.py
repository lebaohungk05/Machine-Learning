"""
Complete pipeline for training and evaluating both models
Run this script to train both CNN and SVM models and generate comparison reports
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_requirements():
    """Check if all required files and directories exist"""
    print_header("Checking Requirements")
    
    # Check data directory
    if not os.path.exists('data/train') or not os.path.exists('data/test'):
        print(" ERROR: Data directory not found!")
        print("Please ensure 'data/train' and 'data/test' directories exist with emotion subdirectories.")
        return False
    
    # Check if models directory exists, create if not
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print(" All requirements satisfied!")
    return True

def run_cnn_training():
    """Train CNN model"""
    print_header("Training CNN Model")
    print("This may take 30-60 minutes depending on your hardware...")
    
    try:
        result = subprocess.run([sys.executable, 'train.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" CNN training completed successfully!")
            return True
        else:
            print(f" CNN training failed with error:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f" Error running CNN training: {e}")
        return False

def run_svm_training():
    """Train SVM+HOG model"""
    print_header("Training SVM+HOG Model")
    print("This should take 10-20 minutes...")
    
    try:
        result = subprocess.run([sys.executable, 'svm_hog.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" SVM training completed successfully!")
            return True
        else:
            print(f" SVM training failed with error:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f" Error running SVM training: {e}")
        return False

def run_analysis():
    """Run comparative analysis"""
    print_header("Running Comparative Analysis")
    
    # Check if result files exist
    if not os.path.exists('models/cnn_results.json') or not os.path.exists('models/svm_results.json'):
        print(" Model results not found! Please train models first.")
        return False
    
    try:
        result = subprocess.run([sys.executable, 'analyze_results.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" Analysis completed successfully!")
            return True
        else:
            print(f" Analysis failed with error:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f" Error running analysis: {e}")
        return False

def print_results_summary():
    """Print summary of results"""
    print_header("Results Summary")
    
    # List generated files
    print(" Generated Files:")
    
    models_files = [
        'models/emotion_model_cnn.h5',
        'models/emotion_model_svm.pkl',
        'models/cnn_results.json',
        'models/svm_results.json',
        'models/confusion_matrix_cnn.csv',
        'models/confusion_matrix_svm.csv'
    ]
    
    plots_files = [
        'plots/training_history_cnn.png',
        'plots/confusion_matrix_cnn.png',
        'plots/confusion_matrix_svm+hog.png',
        'plots/hog_visualization.png',
        'plots/model_comparison.png',
        'plots/comparison_report.html'
    ]
    
    print("\n Models:")
    for file in models_files:
        if os.path.exists(file):
            print(f"   {file}")
        else:
            print(f"   {file} (not found)")
    
    print("\n Visualizations:")
    for file in plots_files:
        if os.path.exists(file):
            print(f"  {file}")
        else:
            print(f"   {file} (not found)")
    
    # Try to load and display key metrics
    try:
        import json
        
        if os.path.exists('models/cnn_results.json'):
            with open('models/cnn_results.json', 'r') as f:
                cnn_results = json.load(f)
            print(f"\n CNN Test Accuracy: {cnn_results['test_accuracy']:.4f}")
        
        if os.path.exists('models/svm_results.json'):
            with open('models/svm_results.json', 'r') as f:
                svm_results = json.load(f)
            print(f" SVM Test Accuracy: {svm_results['test_accuracy']:.4f}")
    except:
        pass
    
    print("\n View the complete comparison report at: plots/comparison_report.html")

def main():
    """Main execution function"""
    print("\n" + " "*20)
    print("  EMOTION RECOGNITION PROJECT - COMPLETE PIPELINE")
    print(" "*20)
    
    start_time = time.time()
    
    # Check requirements
    if not check_requirements():
        print("\n Exiting due to missing requirements.")
        return
    
    # Ask user what to run
    print("\nWhat would you like to do?")
    print("1. Train both models and generate reports (full pipeline)")
    print("2. Train CNN only")
    print("3. Train SVM only")
    print("4. Generate comparison reports only")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        # Full pipeline
        success = True
        if run_cnn_training():
            if run_svm_training():
                run_analysis()
            else:
                success = False
        else:
            success = False
            
    elif choice == '2':
        # CNN only
        run_cnn_training()
        
    elif choice == '3':
        # SVM only
        run_svm_training()
        
    elif choice == '4':
        # Analysis only
        run_analysis()
        
    elif choice == '5':
        print("Exiting...")
        return
    else:
        print("Invalid choice!")
        return
    
    # Print summary
    print_results_summary()
    
    # Print total time
    total_time = time.time() - start_time
    print(f"\n  Total execution time: {total_time/60:.1f} minutes")
    
    print("\n Pipeline completed! Check the plots/ directory for visualizations.")

if __name__ == "__main__":
    main() 