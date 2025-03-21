import os
import cv2
import uuid
import json
import traceback
import numpy as np
from ultralytics import YOLO
from training.samurai import process_video
from utils.polygonHelper import PolygonHelper


model = YOLO('yolo11det.pt')
polyHelper = PolygonHelper()

def bbox_to_polygon(xmin, ymin, xmax, ymax):
    """
    Converts bounding box coordinates (xmin, ymin, xmax, ymax) to a polygon.

    Args:
        xmin (float): The minimum x-coordinate.
        ymin (float): The minimum y-coordinate.
        xmax (float): The maximum x-coordinate.
        ymax (float): The maximum y-coordinate.

    Returns:
        list: A list of points representing the polygon in [x, y] format.
    """
    polygon = [
        [xmin, ymin],  # Bottom-left corner
        [xmax, ymax],  # Top-right corner
        [xmin, ymax],  # Top-left corner
    ]
    return polygon

def get_product_polygon(frame):
    results = model(frame, conf=0.5)
    for result in results:
        bboxes = result.boxes.xyxy.tolist()
    polygons = []
    for box in bboxes:
        box = [int(x) for x in box]
        polygon = bbox_to_polygon(box[0], box[1], box[2], box[3])
        polygons.append(polygon)
    return polygons, bboxes 

def get_rlef_polygons(video_path):
    json_path = video_path.replace('.mp4', '.json')
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def normalize_bbox(bbox, frame_width, frame_height):
    x, y, w, h = bbox
    return [
        x / frame_width,
        y / frame_height,
        w / frame_width,
        h / frame_height
    ]

def compute_features(bbox_sequence):
    features = []
    for i in range(len(bbox_sequence)):
        x, y, w, h = bbox_sequence[i]

        # Normalized bounding box
        norm_bbox = normalize_bbox([x, y, w, h], frame_width, frame_height)

        # Compute displacement, velocity, aspect ratio, and area
        if i > 0:
            prev_x, prev_y, prev_w, prev_h = bbox_sequence[i - 1]
            displacement = [x - prev_x, y - prev_y]
            velocity = np.sqrt(displacement[0]**2 + displacement[1]**2)
        else:
            displacement = [0, 0]
            velocity = 0

        aspect_ratio = w / h if h != 0 else 0
        area = (w * h) / (frame_width * frame_height)

        # Convert all values to Python native types
        feature = norm_bbox + [float(d) for d in displacement] + [float(velocity), float(aspect_ratio), float(area)]
        features.append(feature)
    
    return features

def generate_training_data(dataset_path, video_path, mode, class_name):
    cap = cv2.VideoCapture(video_path)

    # Get video dimensions
    global frame_width, frame_height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print(f"Failed to read first frame for video {video_path}")
        return None

    # _, bboxes = get_product_polygon(first_frame)
    bboxes = get_rlef_polygons(video_path)

    if len(bboxes) == 0:
        print(f"No bounding boxes found in video {video_path}")
        return None

    # Convert bounding boxes to x, y, w, h format
    samurai_bboxes = []
    for box in bboxes:
        x, y, xmax, ymax = box
        w = xmax - x
        h = ymax - y
        samurai_bboxes = [x, y, w, h]

    # Track bounding boxes across frames
    vidname = os.path.basename(video_path)
    sam_save_path = os.path.join("sam2_results", f"track_{vidname}")
    os.makedirs("sam2_results", exist_ok=True)
    _, bbox_sequence = process_video(video_path, samurai_bboxes, model_path="training/samurai/sam2/checkpoints/sam2.1_hiera_large.pt", 
                 save_video=True, output_path=sam_save_path)

    # Compute features for LSTM model
    features = compute_features(bbox_sequence)

    # Create training data structure
    training_data = {
        "class": class_name,
        "features": features
    }

    # Save to JSON file
    json_path = os.path.basename(video_path).replace('.mp4', '.json')
    output_path = os.path.join(dataset_path, class_name, 'json_data', json_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as json_file:
        json.dump(training_data, json_file, indent=4)

    print(f"Training data saved to {output_path}")
    cap.release()
    return output_path

def main(dataset_path, mode):
    class_names = os.listdir(dataset_path)
    for clss in class_names:
        video_dir = os.path.join(dataset_path, clss)
        video_paths = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith('.mp4')]
        
        for video_path in video_paths:
            try:
                generate_training_data(dataset_path, video_path, mode, class_name=clss)
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing video {video_path}: {e}")

if __name__ == "__main__":
    main()
