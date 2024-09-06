import cv2
import torch
import numpy as np

# Path to the trained YOLOv5 model (.pt file)
PATH_TO_TRAINED_MODEL = "C:/Users/vidhy/Documents/PunchBiz .IW/PunchBiz/data/00000001320000000.mp4"

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="C:/Users/vidhy/Documents/PunchBiz .IW/PunchBiz/best (3).pt")

# List to store RoI coordinates
roi_lines = []
roi_regions = []

# Function to handle mouse events for RoI selection
def select_roi(event, x, y, flags, param):
    global roi_lines, roi_regions
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_lines.append((x, y))
    
    elif event == cv2.EVENT_LBUTTONUP:
        roi_lines.append((x, y))
        if len(roi_lines) == 2:
            # Create RoI region based on the coordinates
            roi_start = roi_lines[-2]
            roi_end = roi_lines[-1]
            roi_regions.append((roi_start, roi_end))
            roi_lines = []  # Reset the RoI line coordinates

# Set the mouse event handler for the window
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", select_roi)

def detect_objects_yolov5(image):
    # Perform object detection on the image
    results = model(image)
    
    # Get detected objects and their coordinates
    detected_objects = results.pandas().xyxy[0]
    
    return detected_objects

# Load the video
video_path = "C:/Users/vidhy/Documents/PunchBiz .IW/PunchBiz/data/00000001320000000.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break;
    
    # Create a copy of the frame
    frame_copy = frame.copy()
    
    # Draw the RoI lines on the frame copy
    for roi_start, roi_end in roi_regions:
        cv2.line(frame_copy, roi_start, roi_end, (0, 255, 0), 2)
    
    # Calculate the bounding box that encompasses all RoI lines
    if len(roi_regions) > 0:
        roi_start = np.min([roi_start for (roi_start, _) in roi_regions], axis=0)
        roi_end = np.max([roi_end for (_, roi_end) in roi_regions], axis=0)
        
        # Check if the RoI coordinates are valid
        if roi_start[1] < roi_end[1] and roi_start[0] < roi_end[0]:
            # Detect objects within the RoI
            roi = frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
            detections = detect_objects_yolov5(roi)
            
            # Visualize the detections on the RoI frame
            for _, row in detections.iterrows():
                x_min, y_min, x_max, y_max, confidence, class_id = row["xmin"], row["ymin"], row["xmax"], row["ymax"], row["confidence"], row["class"]
                class_name = model.names[int(class_id)]
                
                cv2.rectangle(frame_copy, (roi_start[0] + int(x_min), roi_start[1] + int(y_min)),
                              (roi_start[0] + int(x_max), roi_start[1] + int(y_max)),
                              (0, 0, 255), 2)
                cv2.putText(frame_copy, f"{class_name}: {confidence:.2f}", (roi_start[0] + int(x_min), roi_start[1] + int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the frame with the RoI lines and object detection
    cv2.imshow("Video", frame_copy)
    
    # Delay for 30 milliseconds (approximately 30 frames per second)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
