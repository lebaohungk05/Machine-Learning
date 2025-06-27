import cv2
import numpy as np
import mediapipe as mp
from model import load_model
from utils import draw_emotion, get_emotion_color

class EmotionDetector:
    def __init__(self, model_path):
        # Load model
        self.model = load_model(model_path)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        # Initialize counters
        self.happy_count = 0
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model input"""
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=[0, -1])
        return face_img
    
    def detect_emotion(self, frame):
        """Detect emotions in the frame"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                # Get face bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Extract and preprocess face
                face_img = frame[y:y+height, x:x+width]
                if face_img.size == 0:
                    continue
                    
                face_processed = self.preprocess_face(face_img)
                
                # Predict emotion
                predictions = self.model.predict(face_processed, verbose=0)[0]
                emotion_idx = np.argmax(predictions)
                emotion = self.emotions[emotion_idx]
                confidence = predictions[emotion_idx]
                
                # Update counters
                self.emotion_counts[emotion] += 1
                if emotion == 'happy':
                    self.happy_count += 1
                
                # Draw results
                color = get_emotion_color(emotion)
                cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display emotion counts
        y_offset = 30
        for emotion, count in self.emotion_counts.items():
            text = f"{emotion}: {count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, get_emotion_color(emotion), 2)
            y_offset += 25
        
        return frame
    
    def reset_counts(self):
        """Reset emotion counters"""
        self.happy_count = 0
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}

def main():
    # Initialize detector
    detector = EmotionDetector("models/emotion_model.h5")
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Thiết lập cửa sổ full màn hình
    cv2.namedWindow('Emotion Detection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Emotion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect emotions
        frame = detector.detect_emotion(frame)
        
        # Display frame
        cv2.imshow('Emotion Detection', frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Reset counters with 'r' key
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            detector.reset_counts()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 