from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folders for uploads and processed videos
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Define the path to your YOLO model file
MODEL_PATH = 'yolov8n.pt'
model = YOLO(MODEL_PATH)
model.fuse()  # Fuse layers to speed up inference


@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process video and get the result
        processed_video_path, num_people = process_video(filepath, filename)

        # Return processed video and number of detected people
        return jsonify({'video_url': processed_video_path, 'people_count': num_people})

    return render_template('index.html')


def process_video(video_path, filename):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up the output video writer
    output_filename = 'processed_' + filename
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    total_persons = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Start processing after the first frame if necessary
        if frame_count > 0:
            # Predict on the current frame
            results = model(frame)

            # Count people and draw bounding boxes
            num_persons = 0
            for *xyxy, conf, cls in results[0].boxes.data:
                if model.names[int(cls)] == 'person':
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    num_persons += 1

            total_persons += num_persons

            # Display the total number of persons detected in this frame
            cv2.putText(frame, f'Total Persons: {num_persons}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Write the frame to the output video
            out.write(frame)

    cap.release()
    out.release()

    return output_filename, total_persons


@app.route('/processed/<filename>')
def download_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
