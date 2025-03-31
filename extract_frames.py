import cv2
import os
import numpy as np
from ultralytics import YOLO

# Define file paths
model_path = "/home/cha0s/motor-alertness/weights/best_body.pt"
video_path = "/home/cha0s/motor-alertness/motor-alertness-dataset/unused_Data/normal/5_5_segment_1.mp4"
output_dir = "saved_frames"

os.makedirs(output_dir, exist_ok=True)

# Load YOLO model
model = YOLO(model_path)

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_counter = 0        # Total number of frames processed
saved_frame_counter = 0  # Counter for naming saved frames

while cap.isOpened():
    ret, orig_frame = cap.read()  # Read the original frame
    if not ret:
        break

    # Keep a copy of the original resolution frame for saving
    original_frame = orig_frame.copy()

    # Resize the frame for faster processing and display
    resized_frame = cv2.resize(orig_frame, (1024, 1024))
    display_frame = resized_frame.copy()

    # Run the model on the resized frame
    results = model(resized_frame)
    largest_box = None

    # Process the detection results
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        classes = result.boxes.cls.cpu().numpy()   # class indices
        person_indices = np.where(classes == 0)[0]   # Assuming 'person' is class 0
        if len(person_indices) == 0:
            continue
        areas = (boxes[person_indices, 2] - boxes[person_indices, 0]) * \
                (boxes[person_indices, 3] - boxes[person_indices, 1])
        largest_index = person_indices[np.argmax(areas)]
        largest_box = boxes[largest_index]

    # Optionally, draw the bounding box on the display frame (currently commented out)
    if largest_box is not None:
        x1, y1, x2, y2 = tuple(map(int, largest_box))
        # Uncomment the lines below to visualize the detection on the resized frame:
        # cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(display_frame, "Largest Person", (x1, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Save every 30th frame (or change the modulo as desired) using the original frame resolution
        if frame_counter % 30 == 0:
            save_path = os.path.join(output_dir, f"person1_frame{saved_frame_counter}.jpg")
            cv2.imwrite(save_path, original_frame)
            print(f"Saved: {save_path}")
            saved_frame_counter += 1

    frame_counter += 1
    cv2.imshow("Video", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
