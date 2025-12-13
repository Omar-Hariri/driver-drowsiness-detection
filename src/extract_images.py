# images
import cv2
import os
import pandas as pd

# --- Paths ---
video_path = r"D:\SPU\5th s1 Lectures\practical Deep Learning\project\data\raw\mirror\Female_mirror\\10-FemaleNoGlasses-Yawning.avi"
labels_csv_path = r"D:\SPU\5th s1 Lectures\Practical Deep Learning\project\data\labels.csv"
output_folder = r"D:\SPU\5th s1 Lectures\Practical Deep Learning\project\data\interim\frames"

# --- Create main output folder ---
os.makedirs(output_folder, exist_ok=True)

# --- Read CSV (no header, 3 columns) ---
df = pd.read_csv(labels_csv_path, header=None)

# --- Filter rows for the current video ---
video_name = os.path.basename(video_path)
subset = df[df[0] == video_name]

# --- Open video ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
   print("Failed to open video.")
   exit()

# --- Extract frames ---
for _, row in subset.iterrows():

   target_time = float(row[1])
   label = row[2]

   # Move video to specific second
   cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)

   ret, frame = cap.read()
   if not ret:
      continue

   # --- Create label folder (if not exist) ---
   label_folder = os.path.join(output_folder, label)
   os.makedirs(label_folder, exist_ok=True)

   # --- Generate clear file name ---
   frame_name = f"{os.path.splitext(video_name)[0]}_{(target_time)}s_{label}.jpg"
   frame_path = os.path.join(label_folder, frame_name)

   # Save frame
   cv2.imwrite(frame_path, frame)

# --- Close video (correct place) ---
cap.release()
cv2.destroyAllWindows()
print("all done ✔")