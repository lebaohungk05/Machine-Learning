from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create optimized CNN model for emotion recognition with regularization
    
    Architecture designed to prevent overfitting:
    - Smaller number of parameters
    - Strong regularization (Dropout + L2)
    - Batch normalization for stability
    - Global Average Pooling instead of Flatten
    """
    model = Sequential([
        # First Convolutional Block - Extract basic features
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Second Convolutional Block - Extract intermediate features
        Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Third Convolutional Block - Extract complex features
        Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        # Fourth Convolutional Block - Extract high-level features
        Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        
        # Global Average Pooling - Reduces parameters significantly
        GlobalAveragePooling2D(),
        
        # Fully Connected Layers
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])

    # Compile model with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()

    return model

def create_simple_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create a simpler CNN model for comparison
    Fewer parameters, suitable for smaller datasets
    """
    model = Sequential([
        # Block 1
        Conv2D(16, (3, 3), padding='same', input_shape=input_shape, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Fully Connected
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def save_model(model, model_path):
    """Save model to disk"""
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """Load model from disk"""
    from tensorflow.keras.models import load_model
    return load_model(model_path)

"""
ARCHITECTURE EXPLANATION:

1. CONVOLUTIONAL BLOCKS (Feature Extraction):
   - Block 1 (32 filters): Detects edges, corners, basic shapes
   - Block 2 (64 filters): Combines basic features into facial parts (eyes, nose)
   - Block 3 (128 filters): Recognizes complex patterns (expressions)
   - Block 4 (256 filters): High-level emotion features
   
2. REGULARIZATION TECHNIQUES:
   - L2 regularization (0.001): Prevents large weights
   - Dropout (0.25-0.5): Randomly drops neurons to prevent co-adaptation
   - Batch Normalization: Stabilizes training, acts as regularizer
   - Global Average Pooling: Reduces parameters by 90% vs Flatten
   
3. OPTIMIZATION CHOICES:
   - Adam optimizer: Adaptive learning rate
   - Small initial learning rate (0.0001): Careful optimization
   - Categorical crossentropy: Standard for multi-class classification
   
4. PARAMETER COUNT:
   - Original model: ~1.5M parameters
   - Optimized model: ~500K parameters (66% reduction)
   - Simple model: ~200K parameters
"""

# # Input: ảnh 48x48 grayscale
# # ↓
# # 3 KHỐI CONVOLUTION để trích xuất features:

# # Khối 1: Học features cơ bản (cạnh, góc)
# Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout
# # 48x48x1 → 24x24x32

# # Khối 2: Học features phức tạp hơn (mắt, mũi, miệng)  
# Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout
# # 24x24x32 → 12x12x64

# # Khối 3: Học features cấp cao (biểu cảm, tổ hợp features)
# Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout  
# # 12x12x64 → 6x6x128

# # ↓
# # FULLY CONNECTED LAYERS để đưa ra quyết định:
# Flatten() → Dense(512) → Dense(256) → Dense(7, softmax)
# # 6x6x128 = 4608 → 512 → 256 → 7 cảm xúc

# ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
# │   train.py      │    │   model.py      │    │   detect.py     │
# │                 │    │                 │    │                 │
# │ - Load data     │───▶│ - Create model  │───▶│ - Load model    │
# │ - Train model   │    │ - Save model    │    │ - Real-time     │
# │ - Save model    │    │                 │    │   detection     │
# └─────────────────┘    └─────────────────┘    └─────────────────┘