from flask import Flask, request, send_file, render_template, jsonify
import os
import subprocess
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "static/results"  # 👈 để frontend truy cập trực tiếp
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def secure_filename(filename):
    return os.path.basename(filename).replace(" ", "_")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    video = request.files['video']
    if not video:
        return "No file uploaded", 400

    # Xử lý tên file gốc
    original_name = secure_filename(video.filename)
    base_name, ext = os.path.splitext(original_name)

    input_path = os.path.join(UPLOAD_FOLDER, original_name)
    result_dir = os.path.join(RESULTS_FOLDER, base_name)
    os.makedirs(result_dir, exist_ok=True)

    # Đường dẫn kết quả
    output_video_path = os.path.join(result_dir, f"{base_name}_tracked.mp4")
    report_path = os.path.join(result_dir, "experiment_report.txt")
    hist1 = os.path.join(result_dir, "object_frequency_histogram.png")
    hist2 = os.path.join(result_dir, "unique_object_histogram.png")
    hist3 = os.path.join(result_dir, "confidence_distribution_histogram.png")

    # Nếu đã xử lý rồi, bỏ qua bước xử lý
    if not os.path.exists(output_video_path):
        video.save(input_path)
        try:
            subprocess.run([
                "python", "detect_with_tracking_2.py",
                "--video_path", input_path,
                "--output", output_video_path
            ], check=True)

            shutil.move("object_frequency_histogram.png", hist1)
            shutil.move("unique_object_histogram.png", hist2)
            shutil.move("confidence_distribution_histogram.png", hist3)
            shutil.move("experiment_report.txt", report_path)

        except subprocess.CalledProcessError as e:
            return f"❌ Script error: {e}", 500

    # Đọc nội dung file txt
    with open(report_path, "r", encoding="utf-8") as f:
        txt_content = f.read()

    return jsonify({
        "video_url": f"/{output_video_path}",
        "histograms": [
            f"/{hist1}",
            f"/{hist2}",
            f"/{hist3}"
        ],
        "report_text": txt_content
    })

if __name__ == '__main__':
    app.run(debug=True)
