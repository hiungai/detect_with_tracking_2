import argparse
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import torch
import os

# === ĐỌC THAM SỐ DÒNG LỆNH ===
parser = argparse.ArgumentParser(description="Chạy thực nghiệm phát hiện & theo dõi đối tượng từ video hành trình")
parser.add_argument('--video_path', type=str, required=True, help='Đường dẫn video đầu vào')
parser.add_argument('--model', type=str, default="yolov8n.pt", help='Tên mô hình YOLO sử dụng')
parser.add_argument('--output', type=str, default="output_tracked.mp4", help='Đường dẫn video đầu ra')
args = parser.parse_args()

video_path = args.video_path
model_path = args.model
save_path = args.output

# === TẢI MÔ HÌNH & TRACKER ===
model = YOLO(model_path)
tracker = DeepSort(max_age=30)

# === MỞ VIDEO INPUT/OUTPUT ===
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

# === BIẾN THỐNG KÊ ===
label_counter = Counter()
object_freq = dict()
class_total_by_object = defaultdict(int)
unique_objects_by_class = defaultdict(set)
confidences = []

# === VÒNG LẶP XỬ LÝ FRAME-BY-FRAME ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    detections = []
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = box
        class_id = int(cls.item())
        class_name = model.model.names[class_id]
        conf_val = conf.item()

        detections.append(([x1.item(), y1.item(), x2.item(), y2.item()], conf_val, class_name))
        label_counter[class_name] += 1
        confidences.append(conf_val)

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        class_name = track.get_det_class()

        if track_id not in object_freq:
            object_freq[track_id] = (class_name, 1)
            class_total_by_object[class_name] += 1
        else:
            prev_class, count = object_freq[track_id]
            object_freq[track_id] = (prev_class, count + 1)

        unique_objects_by_class[class_name].add(track_id)

        color = (0, 255, 0) if class_name == "car" else (0, 0, 255)
        label = f"{class_name} #{track_id}"
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# === THỐNG KÊ KẾT QUẢ CUỐI ===
print("\n=== Tổng số đối tượng duy nhất theo loại ===")
for cls, ids in unique_objects_by_class.items():
    print(f"{cls}: {len(ids)}")

# === VẼ BIỂU ĐỒ HISTOGRAM ===
plt.figure(figsize=(8, 6))
plt.bar(label_counter.keys(), label_counter.values(), color='skyblue')
plt.title("Tổng số lần xuất hiện (frame-based)")
plt.xlabel("Loại đối tượng")
plt.ylabel("Số lần xuất hiện")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("object_frequency_histogram.png")
plt.close()
print("✅ Đã lưu: object_frequency_histogram.png")

plt.figure(figsize=(8, 6))
plt.bar(class_total_by_object.keys(), class_total_by_object.values(), color='orange')
plt.title("Số lượng đối tượng duy nhất theo loại")
plt.xlabel("Loại đối tượng")
plt.ylabel("Số object duy nhất")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("unique_object_histogram.png")
plt.close()
print("✅ Đã lưu: unique_object_histogram.png")

plt.figure(figsize=(8, 6))
plt.hist(confidences, bins=20, color='purple', edgecolor='black')
plt.title("Phân bố độ tin cậy (Confidence) của mô hình")
plt.xlabel("Confidence")
plt.ylabel("Số lượng detection")
plt.tight_layout()
plt.savefig("confidence_distribution_histogram.png")
plt.close()
print("✅ Đã lưu: confidence_distribution_histogram.png")

# === LƯU FILE TXT BÁO CÁO ===
output_txt_path = "experiment_report.txt"
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write("=== KẾT QUẢ THỰC NGHIỆM ===\n")
    f.write(f"Video: {video_path}\n")
    f.write(f"Mô hình: {model_path}\n")
    f.write(f"Tổng số khung hình: {frame_count}\n")
    f.write(f"Kích thước khung hình: {width}x{height}\n")
    f.write(f"Tốc độ khung hình (FPS): {int(fps)}\n")
    f.write("Đối tượng phát hiện:\n")
    for cls, ids in unique_objects_by_class.items():
        f.write(f" - {cls}: {len(ids)} object\n")

    f.write("\n=== THAM SỐ THỰC NGHIỆM ===\n")
    f.write("Confidence threshold: mặc định\n")
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    f.write(f"Thiết bị sử dụng: {device_name}\n")

print("✅ Đã lưu: experiment_report.txt")
