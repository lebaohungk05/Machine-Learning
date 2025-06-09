import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import os

def load_and_preprocess_image(image_path, target_size=(48, 48)):
    """
    Load and preprocess a single image.
    Args:
        image_path: Đường dẫn đến file ảnh cần xử lý
        target_size: Kích thước ảnh đầu ra (mặc định 48x48 pixels)
    Returns:
        Ảnh đã được xử lý hoặc None nếu không đọc được ảnh
    """
    # Đọc ảnh dưới dạng grayscale (ảnh xám)
    # - Sử dụng grayscale vì cảm xúc có thể nhận diện qua cường độ sáng, không cần màu sắc
    # - Giảm kích thước dữ liệu xuống 1/3 so với ảnh RGB
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Kiểm tra nếu không đọc được ảnh
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    # Resize ảnh về kích thước cố định (48x48)
    # - CNN yêu cầu input có kích thước cố định
    # - 48x48 là đủ để nhận diện features của khuôn mặt và cảm xúc
    # - Kích thước nhỏ giúp giảm thời gian training và memory usage
    img = cv2.resize(img, target_size)
    
    # Chuẩn hóa pixel values về khoảng [0,1]
    # - Chuyển từ uint8 (0-255) sang float32 để tính toán chính xác hơn
    # - Chia cho 255 để normalize về khoảng [0,1]
    # - Giúp model học tốt hơn và ổn định hơn
    img = img.astype('float32') / 255.0
    
    return img

def load_dataset(data_dir, target_size=(48, 48), mode='train'):
    """
    Load and preprocess the dataset from directories.
    Args:
        data_dir: Thư mục gốc chứa dữ liệu (data/)
        target_size: Kích thước ảnh đầu ra (mặc định 48x48)
        mode: 'train' hoặc 'test' - chọn tập dữ liệu cần load
    Returns:
        X: Numpy array chứa các ảnh đã xử lý
        y: Numpy array chứa nhãn one-hot encoded
    """
    # Khởi tạo lists rỗng để chứa dữ liệu
    X = []  # Chứa các ảnh
    y = []  # Chứa nhãn (index của cảm xúc)
    
    # Danh sách các cảm xúc - thứ tự này quan trọng vì nó map với one-hot encoding
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    print(f"Loading {mode} images from directories...")
    # Duyệt qua từng loại cảm xúc
    for emotion_idx, emotion in enumerate(emotions):
        # Tạo đường dẫn đến thư mục chứa ảnh của cảm xúc hiện tại
        # Ví dụ: data/train/happy/ hoặc data/test/angry/
        emotion_dir = os.path.join(data_dir, mode, emotion)
        
        # Kiểm tra thư mục có tồn tại không
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory {emotion_dir} does not exist")
            continue
            
        print(f"Loading {emotion} images...")
        # Duyệt qua từng file ảnh trong thư mục
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            # Load và xử lý ảnh
            img = load_and_preprocess_image(img_path, target_size)
            if img is not None:
                X.append(img)  # Thêm ảnh vào list X
                y.append(emotion_idx)  # Thêm index cảm xúc vào list y
    
    if len(X) == 0:
        raise ValueError(f"No {mode} images were loaded. Please check your data directory structure.")
        
    X = np.array(X)
    y = to_categorical(y, num_classes=len(emotions))
    print(f"Loaded {len(X)} {mode} images")
    return X, y



def draw_emotion(frame, face_rect, emotion, confidence):
    """
    Vẽ khung và nhãn cảm xúc lên frame video.
    Args:
        frame: Frame video hiện tại
        face_rect: Tuple (x,y,w,h) chứa tọa độ và kích thước khuôn mặt
        emotion: Tên cảm xúc được dự đoán
        confidence: Độ tin cậy của dự đoán (0-1)
    Returns:
        Frame đã được vẽ
    """
    # Lấy tọa độ khuôn mặt
    x, y, w, h = face_rect
    
    # Vẽ hình chữ nhật xanh lá (BGR: 0,255,0) quanh khuôn mặt
    # Thickness=2 để đường viền dễ nhìn
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Tạo nhãn với format "emotion: 0.XX"
    # .2f để hiển thị 2 chữ số thập phân
    label = f"{emotion}: {confidence:.2f}"
    
    # Vẽ text lên frame, cách viền trên 10px
    # Font HERSHEY_SIMPLEX với scale=0.9 để dễ đọc
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def get_emotion_color(emotion):
    """
    Trả về mã màu BGR tương ứng với mỗi cảm xúc.
    Args:
        emotion: Tên cảm xúc
    Returns:
        Tuple (Blue,Green,Red) đại diện cho màu trong OpenCV
    Note:
        OpenCV sử dụng BGR (Blue,Green,Red) thay vì RGB
    """
    # Dictionary map cảm xúc với màu sắc tương ứng
    # Các màu được chọn để dễ phân biệt trực quan:
    # - Giận dữ: Đỏ - màu của sự nóng giận
    # - Ghê tởm: Xanh lá - màu liên quan đến cảm giác buồn nôn
    # - Sợ hãi: Tím - màu của sự bí ẩn, nguy hiểm
    # - Vui vẻ: Vàng - màu tươi sáng, tích cực
    # - Buồn: Xanh dương - màu lạnh, trầm
    # - Ngạc nhiên: Cam - màu của sự phấn khích
    # - Trung tính: Trắng - màu không có cảm xúc
    colors = {
        'angry': (0, 0, 255),     # Đỏ (BGR)
        'disgust': (0, 128, 0),   # Xanh lá
        'fear': (128, 0, 128),    # Tím
        'happy': (0, 255, 255),   # Vàng
        'sad': (255, 0, 0),       # Xanh dương
        'surprise': (255, 165, 0), # Cam
        'neutral': (255, 255, 255) # Trắng
    }
    # Trả về màu tương ứng, nếu không tìm thấy cảm xúc thì trả về màu trắng
    return colors.get(emotion, (255, 255, 255))

def plot_confusion_matrix(cm, class_names, model_name):
    """
    Vẽ confusion matrix đẹp với heatmap
    
    Args:
        cm: Confusion matrix array
        class_names: List tên các classes
        model_name: Tên model để hiển thị trong title
    """
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix để hiển thị phần trăm
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap với annotations
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2%',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'plots/confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to plots/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.close()

def save_training_report(history, results, model_name):
    """
    Lưu báo cáo training chi tiết dưới dạng HTML
    
    Args:
        history: Training history object
        results: Dictionary chứa kết quả đánh giá
        model_name: Tên model
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{model_name} Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            .section {{ margin: 30px 0; }}
        </style>
    </head>
    <body>
        <h1>{model_name} Model - Training Report</h1>
        
        <div class="section">
            <h2>Overall Performance</h2>
            <p>Test Accuracy: <span class="metric">{results['test_accuracy']:.4f}</span></p>
            <p>Training Time: <span class="metric">{results['training_time_seconds']/60:.2f} minutes</span></p>
        </div>
        
        <div class="section">
            <h2>Model Information</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Model Type</td><td>{results['model_type']}</td></tr>
                <tr><td>Total Parameters</td><td>{results.get('total_parameters', 'N/A'):,}</td></tr>
                <tr><td>Batch Size</td><td>{results.get('batch_size', 'N/A')}</td></tr>
                <tr><td>Learning Rate</td><td>{results.get('learning_rate', 'N/A')}</td></tr>
                <tr><td>Epochs Trained</td><td>{results.get('epochs_trained', 'N/A')}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Per-Class Performance</h2>
            <table>
                <tr><th>Emotion</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>
    """
    
    # Add per-class metrics
    report = results['classification_report']
    for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
        if emotion in report:
            metrics = report[emotion]
            accuracy = results['per_class_accuracy'].get(emotion, 0)
            html_content += f"""
                <tr>
                    <td>{emotion.capitalize()}</td>
                    <td>{accuracy:.4f}</td>
                    <td>{metrics['precision']:.4f}</td>
                    <td>{metrics['recall']:.4f}</td>
                    <td>{metrics['f1-score']:.4f}</td>
                </tr>
            """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Training History</h2>
            <img src="training_history.png" alt="Training History" style="max-width: 100%;">
        </div>
        
        <div class="section">
            <h2>Confusion Matrix</h2>
            <img src="confusion_matrix_""" + model_name.lower().replace(" ", "_") + """.png" alt="Confusion Matrix" style="max-width: 100%;">
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = f'plots/{model_name.lower().replace(" ", "_")}_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Training report saved to {report_path}")

def plot_training_history(history, model_name='Model'):
    """
    Vẽ đồ thị training history với style đẹp hơn
    
    Args:
        history: History object từ model.fit()
        model_name: Tên model để hiển thị
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Style settings
    colors = {'train': '#1f77b4', 'val': '#ff7f0e'}
    
    # Plot accuracy
    epochs = range(1, len(history.history['accuracy']) + 1)
    ax1.plot(epochs, history.history['accuracy'], color=colors['train'], 
             linewidth=2, label='Training Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], color=colors['val'], 
             linewidth=2, label='Validation Accuracy')
    ax1.fill_between(epochs, history.history['accuracy'], alpha=0.2, color=colors['train'])
    ax1.fill_between(epochs, history.history['val_accuracy'], alpha=0.2, color=colors['val'])
    
    ax1.set_title(f'{model_name} - Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot loss
    ax2.plot(epochs, history.history['loss'], color=colors['train'], 
             linewidth=2, label='Training Loss')
    ax2.plot(epochs, history.history['val_loss'], color=colors['val'], 
             linewidth=2, label='Validation Loss')
    ax2.fill_between(epochs, history.history['loss'], alpha=0.2, color=colors['train'])
    ax2.fill_between(epochs, history.history['val_loss'], alpha=0.2, color=colors['val'])
    
    ax2.set_title(f'{model_name} - Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'plots/training_history_{model_name.lower().replace(" ", "_")}.png', 
                dpi=150, bbox_inches='tight')
    print(f"Training history saved to plots/training_history_{model_name.lower().replace(' ', '_')}.png")
    
    plt.show()