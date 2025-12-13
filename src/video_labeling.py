# method 2
import cv2
import keyboard
import time
import os

# --- Video path ---
video_path = r"D:\SPU\5th s1 Lectures\practical Deep Learning\project\data\raw\mirror\Female_mirror\\10-FemaleNoGlasses-Yawning.avi"
video_name = os.path.basename(video_path)

# --- Open video ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
   print("❌ Failed to open video.")
   exit()

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 33

# --- Variables ---
labels = []
current_label = "normal"
paused = False
last_saved_time = -1
label_changed_at = -1   # to avoid saving at the exact change moment

# --- CSV header creation ---
csv_path = "D:\SPU\5th s1 Lectures\practical Deep Learning\project\data\labels.csv"
if not os.path.exists(csv_path):
   with open(csv_path, "w") as f:
      f.write("video_name,second,label\n")


while True:
   if not paused:
      ret, frame = cap.read()
      if not ret:
            print("🎬 End of video reached.")
            break

      current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
      current_sec = int(current_time)

      # --- Dynamic sampling rate ---
      if current_label == "normal":
            interval = 1.0
      else:   # yawn or other
            interval = 0.3

      # --- Save based on dynamic interval ---
      if (current_time - last_saved_time >= interval) and (current_sec != label_changed_at):
            labels.append((video_name, current_time, current_label))
            last_saved_time = current_time
            print(f"{current_time:.2f}s → saved ({current_label})")

      # --- Display video ---
      cv2.putText(frame, f"Label: {current_label}", (30, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      cv2.imshow('Video Labeling', frame)

   # --- Controls ---
   if keyboard.is_pressed('n'):
      current_label = "normal"
      label_changed_at = current_sec
      print(f"→ Label changed to NORMAL")
      time.sleep(0.3)

   elif keyboard.is_pressed('y'):
      current_label = "yawn"
      label_changed_at = current_sec
      print(f"→ Label changed to YAWN")
      time.sleep(0.3)

   elif keyboard.is_pressed(' '):
      paused = not paused
      state = "⏸️ Paused" if paused else "▶️ Resumed"
      print(f"{state} at {current_time:.2f}s")
      time.sleep(0.4)

   elif keyboard.is_pressed('q'):
      print("🛑 Quitting labeling...")
      break

   if not paused:
      if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
   else:
      cv2.waitKey(100)

cap.release()
cv2.destroyAllWindows()

# --- Save labels ---
with open(csv_path, "a") as f:
   for vname, t, label in labels:
      f.write(f"{vname},{t:.2f},{label}\n")

print(f"✅ Labels for video '{video_name}' saved successfully.")
