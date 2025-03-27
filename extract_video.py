import cv2
from pathlib import Path


video_path = Path("./video/balloon.mp4")
cap = cv2.VideoCapture(str(video_path))
frame_save_dir = Path("./test_images/balloon")

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video
    frame_id += 1
    if frame_id < 141:  # ignore first 140 frames
        continue
    frame_save_name = f"{frame_id - 140:08d}.png"
    frame_save_path = frame_save_dir / frame_save_name
    h, w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (int(w/2), int(h/2)))
    cv2.imwrite(str(frame_save_path), resized_frame)
    print(f"Saved: {frame_save_name}")
    

cap.release()
print("Done!")



  
