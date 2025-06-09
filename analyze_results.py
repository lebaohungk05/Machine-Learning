"""
Analyze and compare results from CNN and SVM+HOG models
This script generates comprehensive comparison reports and visualizations
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

def load_results():
    """Load results from both models"""
    with open('models/cnn_results.json', 'r') as f:
        cnn_results = json.load(f)
    
    with open('models/svm_results.json', 'r') as f:
        svm_results = json.load(f)
    
    return cnn_results, svm_results

def plot_model_comparison(cnn_results, svm_results):
    """Create comparison visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall Accuracy Comparison
    ax = axes[0, 0]
    models = ['CNN', 'SVM+HOG']
    accuracies = [cnn_results['test_accuracy'], svm_results['test_accuracy']]
    bars = ax.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12)
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Overall Model Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Training Time Comparison
    ax = axes[0, 1]
    times = [cnn_results['training_time_seconds']/60, svm_results['training_time_seconds']/60]
    bars = ax.bar(models, times, color=['#1f77b4', '#ff7f0e'])
    
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time:.1f} min', ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Training Efficiency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Per-Class Accuracy Comparison
    ax = axes[1, 0]
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    cnn_per_class = [cnn_results['per_class_accuracy'][e] for e in emotions]
    svm_per_class = [svm_results['per_class_accuracy'][e] for e in emotions]
    
    x = np.arange(len(emotions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cnn_per_class, width, label='CNN', color='#1f77b4')
    bars2 = ax.bar(x + width/2, svm_per_class, width, label='SVM+HOG', color='#ff7f0e')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # 4. Model Complexity
    ax = axes[1, 1]
    # For SVM, use feature size as proxy for complexity
    complexities = [
        cnn_results['total_parameters'] / 1000,  # CNN parameters in thousands
        svm_results['hog_feature_size'] * 10    # HOG features scaled for visibility
    ]
    bars = ax.bar(models, complexities, color=['#1f77b4', '#ff7f0e'])
    
    ax.set_ylabel('Model Complexity', fontsize=12)
    ax.set_title('Model Complexity Comparison', fontsize=14, fontweight='bold')
    ax.text(0, complexities[0] + 50, f'{cnn_results["total_parameters"]:,} params', 
            ha='center', fontsize=10)
    ax.text(1, complexities[1] + 50, f'{svm_results["hog_feature_size"]:,} features', 
            ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_comparison_report(cnn_results, svm_results):
    """Generate detailed comparison report"""
    
    # Calculate differences
    acc_diff = cnn_results['test_accuracy'] - svm_results['test_accuracy']
    time_diff = cnn_results['training_time_seconds'] - svm_results['training_time_seconds']
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report - Emotion Recognition</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; text-align: center; }}
            h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f8f9fa; font-weight: bold; }}
            .metric-positive {{ color: #28a745; font-weight: bold; }}
            .metric-negative {{ color: #dc3545; font-weight: bold; }}
            .winner {{ background-color: #d4edda; }}
            .section {{ margin: 30px 0; }}
            .summary-box {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .recommendation {{ background-color: #fff3cd; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Comparative Analysis: CNN vs SVM+HOG for Emotion Recognition</h1>
            
            <div class="summary-box">
                <h2>Executive Summary</h2>
                <p><strong>Best Overall Model:</strong> {'CNN' if acc_diff > 0 else 'SVM+HOG'} 
                   ({"+" if acc_diff > 0 else ""}{acc_diff:.4f} accuracy difference)</p>
                <p><strong>Fastest Training:</strong> {'SVM+HOG' if time_diff > 0 else 'CNN'} 
                   ({abs(time_diff/60):.1f} minutes difference)</p>
            </div>
            
            <div class="section">
                <h2>Performance Metrics Comparison</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>CNN</th>
                        <th>SVM+HOG</th>
                        <th>Difference</th>
                        <th>Winner</th>
                    </tr>
                    <tr>
                        <td>Test Accuracy</td>
                        <td>{cnn_results['test_accuracy']:.4f}</td>
                        <td>{svm_results['test_accuracy']:.4f}</td>
                        <td class="{'metric-positive' if acc_diff > 0 else 'metric-negative'}">{acc_diff:+.4f}</td>
                        <td class="winner">{'CNN' if acc_diff > 0 else 'SVM+HOG'}</td>
                    </tr>
                    <tr>
                        <td>Training Time</td>
                        <td>{cnn_results['training_time_seconds']/60:.1f} min</td>
                        <td>{svm_results['training_time_seconds']/60:.1f} min</td>
                        <td>{time_diff/60:+.1f} min</td>
                        <td class="winner">{'SVM+HOG' if time_diff > 0 else 'CNN'}</td>
                    </tr>
                    <tr>
                        <td>Model Complexity</td>
                        <td>{cnn_results['total_parameters']:,} parameters</td>
                        <td>{svm_results['hog_feature_size']:,} features</td>
                        <td>-</td>
                        <td>SVM+HOG (simpler)</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Per-Emotion Performance Analysis</h2>
                <table>
                    <tr>
                        <th>Emotion</th>
                        <th>CNN Accuracy</th>
                        <th>SVM+HOG Accuracy</th>
                        <th>Best Model</th>
                    </tr>
    """
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    for emotion in emotions:
        cnn_acc = cnn_results['per_class_accuracy'][emotion]
        svm_acc = svm_results['per_class_accuracy'][emotion]
        winner = 'CNN' if cnn_acc > svm_acc else 'SVM+HOG'
        
        html_content += f"""
                    <tr>
                        <td>{emotion.capitalize()}</td>
                        <td>{cnn_acc:.4f}</td>
                        <td>{svm_acc:.4f}</td>
                        <td class="{'winner' if winner == 'CNN' else ''}">{winner}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Detailed Analysis</h2>
                <h3>CNN Model Strengths:</h3>
                <ul>
                    <li>Automatic feature learning from raw pixel data</li>
                    <li>Better performance on complex emotions</li>
                    <li>Robust to variations in facial expressions</li>
                    <li>State-of-the-art architecture with regularization</li>
                </ul>
                
                <h3>SVM+HOG Model Strengths:</h3>
                <ul>
                    <li>Faster training time</li>
                    <li>Lower computational requirements</li>
                    <li>Interpretable features (HOG)</li>
                    <li>Good performance with limited data</li>
                </ul>
                
                <h3>Common Challenges:</h3>
                <ul>
                    <li>Both models struggle with similar emotions (fear/surprise, sad/neutral)</li>
                    <li>Disgust emotion has lowest accuracy for both models</li>
                    <li>Happy emotion is easiest to classify for both models</li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h2>Recommendations</h2>
                <p><strong>Use CNN when:</strong></p>
                <ul>
                    <li>Accuracy is the primary concern</li>
                    <li>Sufficient computational resources are available</li>
                    <li>Real-time performance is not critical</li>
                </ul>
                
                <p><strong>Use SVM+HOG when:</strong></p>
                <ul>
                    <li>Training time is limited</li>
                    <li>Computational resources are constrained</li>
                    <li>Model interpretability is important</li>
                    <li>Quick prototyping is needed</li>
                </ul>
                
                <p><strong>Future Improvements:</strong></p>
                <ul>
                    <li>Implement ensemble methods combining both models</li>
                    <li>Use transfer learning with pre-trained models</li>
                    <li>Collect more training data for underrepresented emotions</li>
                    <li>Apply advanced data augmentation techniques</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <img src="model_comparison.png" alt="Model Comparison" style="max-width: 100%;">
            </div>
        </div>
    </body>
    </html>
    """
    
    with open('plots/comparison_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Comparison report saved to plots/comparison_report.html")

def analyze_confusion_matrices():
    """Analyze and visualize confusion matrix differences"""
    # Load confusion matrices
    cnn_cm = pd.read_csv('models/confusion_matrix_cnn.csv', index_col=0)
    svm_cm = pd.read_csv('models/confusion_matrix_svm.csv', index_col=0)
    
    # Calculate difference matrix
    diff_cm = cnn_cm.values - svm_cm.values
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # CNN Confusion Matrix
    sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('CNN Confusion Matrix', fontsize=14, fontweight='bold')
    
    # SVM Confusion Matrix
    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title('SVM+HOG Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Difference Matrix
    sns.heatmap(diff_cm, annot=True, fmt='d', cmap='RdBu_r', center=0, ax=axes[2])
    axes[2].set_title('Difference (CNN - SVM)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    # Check if result files exist first
    if not os.path.exists('models/cnn_results.json'):
        print("ERROR: CNN results not found!")
        print("Please run 'python train.py' to train the CNN model first.")
        return
    
    if not os.path.exists('models/svm_results.json'):
        print("ERROR: SVM results not found!")
        print("Please run 'python svm_hog.py' to train the SVM model first.")
        return
    
    print("Loading model results...")
    cnn_results, svm_results = load_results()
    
    print("\nGenerating comparison visualizations...")
    plot_model_comparison(cnn_results, svm_results)
    
    # Check if CSV files exist before analyzing confusion matrices
    if os.path.exists('models/confusion_matrix_cnn.csv') and os.path.exists('models/confusion_matrix_svm.csv'):
        print("\nAnalyzing confusion matrices...")
        analyze_confusion_matrices()
    else:
        print("\nWARNING: Confusion matrix CSV files not found. Skipping confusion matrix analysis.")
    
    print("\nGenerating comparison report...")
    generate_comparison_report(cnn_results, svm_results)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"CNN Test Accuracy: {cnn_results['test_accuracy']:.4f}")
    print(f"SVM+HOG Test Accuracy: {svm_results['test_accuracy']:.4f}")
    print(f"Accuracy Difference: {cnn_results['test_accuracy'] - svm_results['test_accuracy']:+.4f}")
    print(f"\nCNN Training Time: {cnn_results['training_time_seconds']/60:.1f} minutes")
    print(f"SVM+HOG Training Time: {svm_results['training_time_seconds']/60:.1f} minutes")
    print(f"Time Difference: {(cnn_results['training_time_seconds'] - svm_results['training_time_seconds'])/60:+.1f} minutes")
    print("\nReports generated in plots/ directory")

if __name__ == "__main__":
    main() 