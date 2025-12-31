import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

import winsound  # for alert
import time
import os

# -------------------------------
# Load Trained Model
# -------------------------------
model = tf.keras.models.load_model(r"D:\SPU\5th s1 Lectures\Practical Deep Learning\model_b\model.h5")
# model = tf.keras.models.load_model(r"../models/cnn_v3/best.keras")

# Optional: class labels
CLASSES = {0: "NORMAL", 1: "YAWN"}

# -------------------------------
# Initialize MediaPipe Face Mesh
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
   static_image_mode=False,
   max_num_faces=1,
   refine_landmarks=True,
   min_detection_confidence=0.5,
   min_tracking_confidence=0.5
)

# -------------------------------
# Prediction Function
# -------------------------------
def predict_face_state(face_img, model):
   try:
      # Resize to 64x64
      face_img = cv2.resize(face_img, (64, 64))
      # Convert to grayscale
      face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
      # Normalize
      face_img = face_img.astype("float32") / 255.0
      # Add batch & channel dimension
      face_img = np.expand_dims(face_img, axis=(0, -1))  # (1,64,64,1)
      # Predict
      pred = model.predict(face_img, verbose=0)
      score = float(pred[0][0])
      predicted_class = 1 if score >= 0.5 else 0
      confidence = score if predicted_class == 1 else 1 - score
      return predicted_class, confidence
   except Exception as e:
      print("Prediction Error:", e)
      return None, 0.0

# -------------------------------
# Helper: get square bbox from landmarks
# -------------------------------
def get_face_bbox(landmarks, img_w, img_h, padding=20):
   xs = [int(lm.x * img_w) for lm in landmarks]
   ys = [int(lm.y * img_h) for lm in landmarks]
   x1, x2 = max(min(xs) - padding, 0), min(max(xs) + padding, img_w)
   y1, y2 = max(min(ys) - padding, 0), min(max(ys) + padding, img_h)
   return x1, y1, x2, y2

# -------------------------------
# Main Loop
# -------------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
   ret, frame = cap.read()
   if not ret:
      break

   # frame = cv2.flip(frame, 1)
   h, w, _ = frame.shape

   # Convert to RGB for MediaPipe
   rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   results = face_mesh.process(rgb_frame)

   overall_status = "UNKNOWN"
   overall_color = (255, 255, 255)

   if results.multi_face_landmarks:
      # Only process first detected face
      face_landmarks = results.multi_face_landmarks[0]
      x1, y1, x2, y2 = get_face_bbox(face_landmarks.landmark, w, h, padding=20)
      face_crop = frame[y1:y2, x1:x2]

      if face_crop.size > 0:
            pred_class, conf = predict_face_state(face_crop, model)
            if pred_class is not None:
               overall_status = CLASSES.get(pred_class, "UNKNOWN")
               overall_color = (0, 255, 0) if overall_status == "NORMAL" else (0, 0, 255)
               if overall_status == "YAWN":
                  winsound.Beep(1000, 500)  # optional alert

               # Draw bounding box & label
               cv2.rectangle(frame, (x1, y1), (x2, y2), overall_color, 2)
               cv2.putText(frame, f"{overall_status} {int(conf*100)}%", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, overall_color, 2)

   # Display overall status
   cv2.putText(frame, f"Status: {overall_status}", 
               (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, overall_color, 2)

   # Show frame
   cv2.imshow("Driver Drowsiness Detection", frame)

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
