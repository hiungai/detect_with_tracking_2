📄 readme.txt — Hướng Dẫn Sử Dụng Hệ Thống Phát Hiện & Theo Dõi Đối Tượng

📌 Mô tả:
Đây là hệ thống web sử dụng mô hình YOLO kết hợp DeepSORT để phát hiện và theo dõi các đối tượng (xe, người,...) trong video hành trình. Kết quả gồm:
- Video đã gắn nhãn & theo dõi
- 3 biểu đồ histogram trực quan
- Báo cáo thống kê .txt

🛠 Yêu cầu cài đặt:
- Python 3.8+
- Các thư viện:
  pip install flask opencv-python ultralytics deep_sort_realtime matplotlib torch

📁 Cấu trúc thư mục:
project/
├── app.py                        # Flask web server
├── detect_with_tracking_2.py    # Script xử lý video
├── uploads/                     # Nơi lưu video gốc
├── static/
│   └── results/                 # Kết quả đầu ra theo từng video
├── templates/
│   └── index.html               # Giao diện web

🚀 Cách chạy ứng dụng:
1. Mở terminal tại thư mục chứa app.py
2. Chạy Flask server:
   python app.py
3. Truy cập giao diện web:
   http://127.0.0.1:5000/

📤 Cách sử dụng:
1. Truy cập trang web
2. Tải lên video .mp4 hành trình
3. Nhấn “Tải lên & Xử lý”
4. Chờ xử lý hoàn tất:
   - Xem video kết quả
   - Xem 3 biểu đồ histogram
   - Xem báo cáo thống kê chi tiết

👉 Nếu bạn upload một video đã xử lý trước đó, hệ thống sẽ trả về kết quả cũ ngay mà không cần xử lý lại.

📦 Kết quả xuất ra:
Sau mỗi lần xử lý, hệ thống tạo thư mục theo tên video gốc trong static/results/, ví dụ:
static/results/video/
├── video_tracked.mp4
├── object_frequency_histogram.png
├── unique_object_histogram.png
├── confidence_distribution_histogram.png
└── experiment_report.txt

❓ Liên hệ hỗ trợ:
- Người phát triển: Ngô Nguyễn Hiếu Nghĩa
- Trường: Đại học Cần Thơ
- Môn học: Business Intelligence (Nghiệp vụ thông minh)

✅ Chúc bạn sử dụng hệ thống hiệu quả!
