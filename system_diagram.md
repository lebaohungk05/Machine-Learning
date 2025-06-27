```mermaid
graph LR
    A[Input Video<br/>Webcam] --> B[Phát hiện khuôn mặt<br/>MediaPipe Face Detection]
    B --> C[Trích xuất & Tiền xử lý<br/>- Cắt khuôn mặt<br/>- Resize 48x48<br/>- Chuẩn hóa pixel]
    C --> D[Phân loại cảm xúc<br/>CNN Model]
    D --> E[Hiển thị kết quả<br/>- Cảm xúc<br/>- Độ tin cậy]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#fbb,stroke:#333,stroke-width:2px
    style E fill:#fbf,stroke:#333,stroke-width:2px
```

# Sơ đồ khối tổng quan của hệ thống nhận diện cảm xúc

## Mô tả các thành phần:

1. **Input Video (Webcam)**
   - Nguồn dữ liệu đầu vào từ webcam
   - Cung cấp luồng video real-time

2. **Phát hiện khuôn mặt (MediaPipe)**
   - Sử dụng MediaPipe Face Detection
   - Xác định vị trí khuôn mặt trong frame
   - Trả về tọa độ bounding box

3. **Trích xuất & Tiền xử lý**
   - Cắt khuôn mặt từ frame gốc
   - Resize về kích thước 48x48 pixels
   - Chuẩn hóa giá trị pixel về khoảng [0,1]

4. **Phân loại cảm xúc (CNN Model)**
   - Mô hình CNN đã được huấn luyện
   - Phân loại 7 cảm xúc cơ bản
   - Trả về nhãn cảm xúc và độ tin cậy

5. **Hiển thị kết quả**
   - Hiển thị cảm xúc được nhận diện
   - Hiển thị độ tin cậy của dự đoán
   - Cập nhật real-time theo từng frame 