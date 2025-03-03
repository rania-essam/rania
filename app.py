import cv2
import os
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from moviepy import VideoFileClip
import librosa
from joblib import load
from flask import Flask, request, jsonify
app = Flask(__name__)
input_video = r"test_lie.mp4"
saved_model = load_model(r"lstm_without_masking.h5")

model = load_model("audio_model.h5")
# check if the input video has no person or has multiple persons
# returns True if the input video is correct , false otherwise
def input_validation(input_video):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("Could not open video")

    valid = True
    while cap.isOpened():
        ret,frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        if not results.detections or len(results.detections) > 1:
            valid = False
            break

    cap.release()

    return valid

def check_audio_in_video(video_file):
    """Check if the video contains an audio track."""
    video = VideoFileClip(video_file)
    if video.audio is None:
        return 0
    return 1

mp_face_detection = mp.solutions.face_detection
def crop_face_from_frame(frame):
    height, width, _ = frame.shape
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * width)
            y = int(bboxC.ymin * height)
            w = int(bboxC.width * width)
            h = int(bboxC.height * height)

            padding = 0.2
            pad_x = int(padding * w)
            pad_y = int(padding * h)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(width - x, w + 2 * pad_x)
            h = min(height - y, h + 2 * pad_y)

            cropped_face = frame[y:y+h, x:x+w]
            cropped_face_resized = cv2.resize(cropped_face, (width, height))
            return cropped_face_resized
    return None

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,  # to tell the model it is a video not image
        max_num_faces=1,   # number of faces the model should detect
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True  # add points for gaze tracking
    )

def mediapipe_predictions(frame,model):
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = face_mesh.process(frame)
  return results

def extract_frame_points(face_landmarks):
  frame_points=[]
  for landmark in face_landmarks.landmark:
    x = landmark.x
    y = landmark.y
    z = landmark.z
    frame_points.append(x)
    frame_points.append(y)
    frame_points.append(z)

  return frame_points

def apply_facemesh(frame):
  results=mediapipe_predictions(frame,face_mesh)
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      frame_points=extract_frame_points(face_landmarks)
  else:
    frame_points=np.zeros(478*3)
  return frame_points

labels=['truth','lie']
def make_prediction(input_video):
    global saved_model

    # Open video capture
    video_capture = cv2.VideoCapture(input_video)
    all_video_predictions = []

    if not video_capture.isOpened():
        return {"error": "Could not open video"}

    valid_input = input_validation(input_video)
    if not valid_input:
        return {"error": "The input video should have only one person with a clearly visible face"}

    frame_count = 0
    cur_video32 = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1

        cropped_frame = crop_face_from_frame(frame)
        if cropped_frame is None:
            cur_video32.append(np.zeros(478 * 3))  # Add zeros if no face detected
        else:
            frame_points = apply_facemesh(cropped_frame)
            cur_video32.append(frame_points)

        if frame_count == 32:
            input_data = np.array([cur_video32])
            prediction = saved_model.predict(input_data) if saved_model else [0]  # Dummy prediction
            

           
            all_video_predictions.append(np.argmax(prediction))
            

            cur_video32 = []
            frame_count = 0

    # Handle remaining frames
    if frame_count != 0:
        while frame_count < 32:
            cur_video32.append(np.zeros(478 * 3))
            frame_count += 1
        input_data = np.array([cur_video32])
        prediction = saved_model.predict(input_data) if saved_model else [0]  # Dummy prediction

       

        
        all_video_predictions.append(np.argmax(prediction))
        

    return  all_video_predictions
def extract_audio_from_video(video_file, audio_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(audio_file, codec='pcm_s16le')

import librosa
import numpy as np

def split_and_pad_audio(audio_file, segment_duration=10, sample_rate=16000):
    segment_samples = segment_duration * sample_rate

    audio, sr = librosa.load(audio_file, sr=sample_rate)

    num_segments = len(audio) // segment_samples
    remainder = len(audio) % segment_samples

    segments = []
    for i in range(num_segments):
        segment = audio[i * segment_samples : (i + 1) * segment_samples]
        segments.append(segment)

    if remainder > 0:
        last_segment = audio[num_segments * segment_samples:]
        padded_segment = np.pad(last_segment, (0, segment_samples - len(last_segment)), mode='constant')
        segments.append(padded_segment)

    return segments

def extract_mfcc_from_segments(segments, sample_rate=16000, n_mfcc=13):
    mfcc_features = []

    for idx, segment in enumerate(segments):
        mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        feature_entry = {
            "segment_index": idx + 1,
        }
        for i in range(n_mfcc):
            feature_entry[f"mfcc{i+1}_mean"] = mfcc_mean[i]
            feature_entry[f"mfcc{i+1}_std"] = mfcc_std[i]

        mfcc_features.append(feature_entry)

    return pd.DataFrame(mfcc_features)

def make_prediction_audio(input_video):
  video_capture = cv2.VideoCapture(input_video)
  all_video_predictions=[]
  if not video_capture.isOpened():
    return('Could not open video')
  valid_input_voice=check_audio_in_video(input_video)
  if valid_input_voice == False:
     return ("the input video should have audio")
  audio_file = "extracted_audio.wav"
  extract_audio_from_video(input_video, audio_file)
  segments = split_and_pad_audio(audio_file, segment_duration=10, sample_rate=16000)
  mfcc_data_df = extract_mfcc_from_segments(segments, sample_rate=16000, n_mfcc=13)
  mfcc_data_df.drop(['segment_index'], axis=1, inplace=True)
  scaler = load(r"scaler.joblib" )
  mfcc_data_df = scaler.transform(mfcc_data_df)
  mfcc_data_reshaped = np.expand_dims(mfcc_data_df, axis=1)
  y_pred = model.predict(mfcc_data_reshaped)
  voice_predictions = np.argmax(y_pred, axis=1)
  voice_predictions = 1 - voice_predictions
  return voice_predictions

import os
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

latest_result = None

def final_decision(input_video):
    global latest_result

    # Get voice and facial predictions
    voice_predictions = make_prediction_audio(input_video)
    facial_predictions = make_prediction(input_video)

    # Ensure voice predictions match facial predictions length
    # Repeat voice predictions (you may adjust the factor here)
    replicated_voice_predictions = np.repeat(voice_predictions, 10, axis=0)

    # If replicated voice predictions are longer than facial predictions, trim them
    if len(replicated_voice_predictions) > len(facial_predictions):
        replicated_voice_predictions = replicated_voice_predictions[:len(facial_predictions)]
    # If replicated voice predictions are shorter than facial predictions, pad them
    elif len(replicated_voice_predictions) < len(facial_predictions):
        replicated_voice_predictions = np.pad(replicated_voice_predictions, (0, len(facial_predictions) - len(replicated_voice_predictions)), constant_values=1)

    # Ensure both are NumPy arrays
    replicated_voice_predictions = np.array(replicated_voice_predictions)
    facial_predictions = np.array(facial_predictions)

    # Hard voting decision
    hard_vote = (replicated_voice_predictions + facial_predictions) >= 1
    hard_vote = hard_vote.astype(int)

    lie = np.sum(hard_vote)
    truth = len(hard_vote) - lie
    confidence = (max(lie, truth) / len(hard_vote)) * 100
    final_result = "Lie" if lie > truth else "Truth"

    latest_result = {
        "voice_predictions": replicated_voice_predictions.tolist(),
        "facial_predictions": facial_predictions.tolist(),
        "hard_vote": hard_vote.tolist(),
        "final_decision": final_result,
        "confidence": confidence
    }
    return latest_result


@app.route("/process_video", methods=["POST"])
def process_video():
    global latest_result

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]

    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], video_file.filename)
    video_file.save(file_path)

    result = final_decision(file_path)  # Call final_decision to process the video
    latest_result = result  # Store latest result

    return jsonify(result)


@app.route("/result", methods=["GET"])
def get_result():
    if latest_result is None:
        return jsonify({"error": "No analysis has been performed yet"}), 404
    return jsonify(latest_result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
