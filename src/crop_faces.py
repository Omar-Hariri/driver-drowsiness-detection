import cv2
import mediapipe as mp
import os
from tqdm import tqdm
# to test and show some faces
import os
# paths
input_dir = r"D:\SPU\5th s1 Lectures\Practical Deep Learning\project\data\interim\frames"
output_dir = r"D:\SPU\5th s1 Lectures\Practical Deep Learning\project\data\processed\faces"

# initialize mediapipe face detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# loop over labels (subfolders)
for label in os.listdir(input_dir):
   label_input_path = os.path.join(input_dir, label)
   label_output_path = os.path.join(output_dir, label)
   
   if not os.path.exists(label_output_path):
      os.makedirs(label_output_path)
   
   image_files = [f for f in os.listdir(label_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
   
   for img_name in tqdm(image_files, desc=f"Processing {label}"):
      img_path = os.path.join(label_input_path, img_name)
      img = cv2.imread(img_path)
      if img is None:
            continue
      
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      results = face_detection.process(img_rgb)
      
      if results.detections:
            # take the first detected face
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            x1 = max(int(bboxC.xmin * w), 0)
            y1 = max(int(bboxC.ymin * h), 0)
            x2 = min(int((bboxC.xmin + bboxC.width) * w), w)
            y2 = min(int((bboxC.ymin + bboxC.height) * h), h)
            
            face_crop = img[y1:y2, x1:x2]
            
            out_path = os.path.join(label_output_path, img_name)
            cv2.imwrite(out_path, face_crop)
