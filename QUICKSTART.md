# Quick Start Guide

##  Fastest Way to Get Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Ensure your dataset follows this structure:
```
data/
├── train/
│   ├── angry/     (contains angry face images)
│   ├── disgust/   (contains disgust face images)
│   ├── fear/      (contains fear face images)
│   ├── happy/     (contains happy face images)
│   ├── sad/       (contains sad face images)
│   ├── surprise/  (contains surprise face images)
│   └── neutral/   (contains neutral face images)
└── test/
    └── (same structure as train)
```

### 3. Run Everything Automatically
```bash
python run_all.py
```
Then select option 1 to train both models and generate all reports.

### 4. View Results
After completion, check:
- `plots/comparison_report.html` - Complete analysis report
- `plots/` directory - All visualizations
- `models/` directory - Saved models and metrics

##  Individual Commands

### Train CNN Only
```bash
python train.py
```

### Train SVM Only
```bash
python svm_hog.py
```

### Generate Comparison Report
```bash
python analyze_results.py
```

### Real-time Emotion Detection
```bash
python detect.py
```

##  Tips for Better Results

1. **More Data = Better Performance**
   - CNN needs at least 1000 images per emotion
   - SVM can work with fewer samples

2. **Balanced Dataset**
   - Ensure similar number of images per emotion
   - Use data augmentation for underrepresented classes

3. **Hardware Recommendations**
   - GPU recommended for CNN training
   - At least 8GB RAM
   - SSD for faster data loading

##  Common Issues

### Out of Memory
- Reduce batch size in `train.py` (line with `batch_size=32`)
- Close other applications

### Slow Training
- Use GPU if available
- Reduce epochs for quick testing
- Start with SVM for faster results

### Poor Accuracy
- Check data quality (faces should be centered)
- Increase data augmentation
- Try transfer learning approach 