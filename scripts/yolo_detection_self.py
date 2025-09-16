#!/usr/bin/env python3

import cv2
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import time
from datetime import datetime
import pytz
import yaml
import requests


def post_json_data(json_data, post_url, timeout=10):
    """POST JSON data to specified endpoint."""
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(post_url, json=json_data, headers=headers, timeout=timeout)
        response.raise_for_status()
        print(f"Successfully posted JSON data to {post_url}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to post JSON data to {post_url}: {e}")
        return False


def detection(detector, frame, json_tmp, conf_threshold,class_list):
    results = detector.predict(frame, conf=conf_threshold, verbose=False, classes=class_list)
    detections,image_list = [], []
    
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                # Get class name from model
                class_name = detector.names.get(class_id, 'unknown')
                detections.append({"x1": x1,
                                   "y1": y1, 
                                   "x2": x2,
                                   "y2": y2,
                                   "width": x2 - x1,
                                   "height": y2 - y1
                                   })
                bounded_img=frame.copy()         
                cv2.rectangle(bounded_img, (x1, y1), (x2, y2), (0, 255, 0), 2)       
                image_list.append(bounded_img)    
    json_tmp.update({"bounding_box": detections
                     },)
    return json_tmp,image_list


def load_model_config(config_path="./config.yaml"):
    """Load model configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

def get_config(config_type, key, config):
    """Get model path from config based on model type."""
    config_list = config.get(config_type, {})
    if key not in config_list:
        available_models = list(config_list.keys())
        print(f"Model type '{key}' not found in config.")
        print(f"Available models: {available_models}")
        sys.exit(1)
    return config_list[key]

def get_time():
    hong_kong_tz = pytz.timezone('Asia/Hong_Kong')
    now = datetime.now(hong_kong_tz)

    # Extract components
    day = now.strftime('%d')
    month = now.strftime('%m')
    year = now.strftime('%Y')
    hour = now.strftime('%H')
    minute = now.strftime('%M')
    second = now.strftime('%S')

    return day,month,year,hour,minute,second

def create_output_directories():
    day, month, year, hour, minute, second = get_time()
    base_path="./output"
    
    # Create directory path
    dir_path = Path(base_path) / year / month / day / hour
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for images and json
    images_dir = dir_path / "images"
    json_dir = dir_path / "json"
    images_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)
    
    return images_dir, json_dir

def rtsp_stream_init(rtsp_url):
    # Initialize RTSP stream with timeout settings
    video_cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Optimize buffer settings for low latency and set timeout
    video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS
    # Set RTSP timeout to 60 seconds (in milliseconds)
    video_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
    video_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)
    # Additional optimizations
    video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return video_cap

def blur_face(image,frame_count):
    face_detector = YOLO("./models/face_bounding.pt")
    results = face_detector.predict(image, conf=0.7, verbose=False)
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Apply Gaussian blur to the detected face region
                face_region = image[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                image[y1:y2, x1:x2] = blurred_face
                print(f"Blurred face at frame {frame_count}")
    return image


def main():
    # Load configuration
    config = load_model_config()
    
    # Get API configuration
    api_config = config.get("api", {})
    post_endpoint = api_config.get("post_endpoint", "http://post-server:8080/api/detections")
    api_timeout = api_config.get("timeout", 10)
    
    # Create output directories
    images_dir, json_dir = create_output_directories()
    print(f"Output directories created:")
    print(f"Images: {images_dir}")
    print(f"JSON: {json_dir}")
    print(f"POST endpoint: {post_endpoint}")
    
    # Get configuration from environment variables
    model_type = os.getenv('YOLO_TYPE', 'person')
    stream_url = os.getenv('YOLO_STREAM', 'rtsp://18.167.218.143:10554/34020000001320118007_34020000001320118007')
    blur_enabled = os.getenv('YOLO_BLUR', 'true').lower() == 'true'

    video_cap = rtsp_stream_init(stream_url)
    print(f"Starting RTSP stream processing: {stream_url}")
    print("Press 'q' to quit")

    # Load model based on model_type from environment using config
    model_path = get_config("models", model_type, config)
    detector = YOLO(model_path)
    
    frame_count = 0
    
    try:
        while True:
            success, frame = video_cap.read()
            if success:
                frame_count += 1
                if frame_count % 10 == 0:
                    detection_tmp = {
                        "model_type": model_type,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "robot": get_config("robot", stream_url, config),
                        "camera": get_config("camera", stream_url, config),
                        "pose":  {"x" : 0,
                                  "y" : 0,
                                  "z" : 0,
                                  "r" : 0,
                                  "p" : 0,
                                  "y" : 0,}
                    }
                    if model_type in ["bicylce",
                                      "pets",
                                      "vehicle",
                                      "person"]:
                        class_list = get_config("classes", model_type, config)
                    else:
                        class_list=[0]


                    # Get detections as JSON array
                    frame_detection, bounded_images = detection(detector, frame, detection_tmp, get_config("confidence", model_type, config),class_list)

                    img_path_list=[]
                    for i,img in enumerate(bounded_images):
                        img_filename = f"detection_{model_type}_{frame_count}_obj_{i}.jpg"
                        img_filepath = images_dir / img_filename
                        if blur_enabled:
                            img = blur_face(img,frame_count)
                        cv2.imwrite(str(img_filepath), img)
                        img_path_list.append(str(img_filepath))


                    # Save individual frame detection JSON
                    json_filename = f"detection_frame_{frame_count}.json"
                    json_filepath = json_dir / json_filename
                    frame_detection.update({"image path": img_path_list})
                    with open(json_filepath, 'w') as f:
                        json.dump(frame_detection, f, indent=2)
                    
                    # POST the JSON data
                    post_json_data(frame_detection, post_endpoint, api_timeout)
                    print(f"Frame {frame_count}: {len(bounded_images)} detections saved")

                    # # Display frame with bounding boxes
                    # if bounded_images:
                    #     cv2.imshow("RTSP Detection", bounded_images[0])
                    # else:
                    #     cv2.imshow("RTSP Detection", frame)

                    # # Reduced wait time for better performance
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
            else:
                print("Failed to read frame from RTSP stream")
                # Try to reconnect with proper settings
                video_cap.release()
                time.sleep(1)
                video_cap = rtsp_stream_init(stream_url)
                if not video_cap.isOpened():
                    print("Failed to reconnect to RTSP stream")
                    break
                    
    except KeyboardInterrupt:
        print("\nStopping RTSP stream processing...")
    finally:
        print(f"Total frames processed: {frame_count}")
        # Cleanup
        video_cap.release()

if __name__=="__main__":
    main()