#!/usr/bin/env python

import datetime
import cv2
import numpy as np
from datetime import datetime, timedelta
import base64
import json
import urllib.request
import time
import logging
from logging.handlers import RotatingFileHandler
import os
import argparse
import sys
import pytz
import yaml
from pathlib import Path
from ultralytics import YOLO
from tcp_listener import TCP_Listener

# Set FFMPEG environment variables for stable RTSP streaming with RKNN hardware acceleration
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|buffer_size;1024000|max_delay;500000|hwaccel;rkmpp'
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '1'
os.environ['OPENCV_FFMPEG_ENABLE_MPP'] = '1'


def setup_logger():
    """Configure logging with rotation"""
    logger = logging.getLogger('object_detection')
    logger.setLevel(logging.DEBUG)

    handler = RotatingFileHandler(
        'object_detection_new.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


logger = setup_logger()

# Global variables for 24-hour reset
uploaded_ids = {}  # Format: {id: timestamp}
last_reset_time = datetime.now()
RESET_INTERVAL = timedelta(hours=24)


def can_upload_id(track_id):
    """Check if ID can be uploaded based on 24-hour reset"""
    current_time = datetime.now()

    if track_id not in uploaded_ids:
        uploaded_ids[track_id] = current_time
        return True

    if (current_time - uploaded_ids[track_id]) >= RESET_INTERVAL:
        uploaded_ids[track_id] = current_time
        return True

    return False


def check_and_reset_ids():
    """Check and reset ID system if 24 hours have passed"""
    global last_reset_time, uploaded_ids
    current_time = datetime.now()

    if (current_time - last_reset_time) >= RESET_INTERVAL:
        logger.info("Performing 24-hour reset of tracking system")
        uploaded_ids.clear()
        last_reset_time = current_time
        return True
    return False


def send_detection_to_server(frame, detections, server_url, device_id, vin, picture_type):
    """Send detection with new endpoint structure"""
    try:
        logger.info(f"Preparing to send to endpoint: {server_url}")
        logger.info(f"Device ID: {device_id}, VIN: {vin}, Picture Type: {picture_type}")
        logger.info(f"Number of detections: {len(detections)}")

        # Encode image to base64
        retval, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)

        # Prepare message with new structure
        # Format time as yyyy-mm-dd hh:mm:ss
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = {
            "vin": vin,
            "deviceId": device_id,
            "pictureType": picture_type,
            "capTime": now,
            "longitude": float(0),
            "latitude": float(0),
            "picture": jpg_as_text.decode('utf-8'),
            "detections": detections
        }

        logger.info(f"Message payload prepared (picture size: {len(jpg_as_text)} bytes)")

        # Log message without picture field
        msg_without_picture = {
            "vin": vin,
            "deviceId": device_id,
            "pictureType": picture_type,
            "capTime": now,
            "longitude": float(0),
            "latitude": float(0),
            "detections": detections
        }
        logger.info(f"Output message (without picture): {json.dumps(msg_without_picture, indent=2)}")
        print(f"Output message (without picture): {json.dumps(msg_without_picture, indent=2)}")

        # Send request
        json_msg = json.dumps(msg).encode('UTF-8')
        req = urllib.request.Request(server_url, json_msg)
        req.add_header('Content-Type', 'application/json')

        logger.info(f"Sending POST request to {server_url}")
        response = urllib.request.urlopen(req)
        res = json.loads(response.read().decode('UTF-8'))

        # Log server response with status
        logger.info(f"Server response status: {response.status}")
        logger.info(f"Server response body: {res}")
        print(f"Server response status: {response.status}")
        print(f"Server response body: {res}")

        return True

    except Exception as e:
        logger.error(f"Error sending detection to server: {e}")
        print(f"Error sending detection to server: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def detection(detector, frame, conf_threshold, class_list, track_counts, posted_track_ids, track_threshold=5):
    """Perform detection with tracking"""
    results = detector.track(frame, conf=conf_threshold, verbose=False, classes=class_list, persist=True)
    detections = []
    bbox_list = []
    track_ids_to_post = []
    frame_height, frame_width = frame.shape[:2]

    # Track which IDs are present in current frame
    current_track_ids = set()

    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Get track ID if available
                track_id = None
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0])
                    current_track_ids.add(track_id)

                    # Update consecutive count
                    if track_id not in track_counts:
                        track_counts[track_id] = 0
                    track_counts[track_id] += 1

                    # Check if this track should be posted
                    if track_counts[track_id] >= track_threshold and track_id not in posted_track_ids:
                        track_ids_to_post.append(track_id)

                # Filter out full-frame detections
                if (x2 - x1) * (y2 - y1) / (frame_height * frame_width) < 0.9:
                    if track_id is not None and track_id in track_ids_to_post:
                        detections.append({
                            "id": track_id,
                            "bbox": [x1, y1, x2, y2]
                        })
                        bbox_list.append([x1, x2, y1, y2])

    # Reset count for tracks that disappeared
    tracks_to_reset = [tid for tid in track_counts if tid not in current_track_ids and tid not in posted_track_ids]
    for tid in tracks_to_reset:
        track_counts[tid] = 0

    return detections, bbox_list, track_ids_to_post


def load_model_config(config_path="./config.yaml"):
    """Load model configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        print(f"Error parsing config file: {e}")
        sys.exit(1)


def get_config(config_type, key, config):
    """Get configuration value from config"""
    config_list = config.get(config_type, {})
    if key not in config_list:
        available_models = list(config_list.keys())
        logger.error(f"Key '{key}' not found in config type '{config_type}'. Available: {available_models}")
        print(f"Key '{key}' not found in config type '{config_type}'.")
        print(f"Available: {available_models}")
        sys.exit(1)
    return config_list[key]


def get_time(tz):
    """Get current time components"""
    now = datetime.now(tz)
    day = now.strftime('%d')
    month = now.strftime('%m')
    year = now.strftime('%Y')
    hour = now.strftime('%H')
    return day, month, year, hour


def create_output_directories(tz):
    """Create output directory structure"""
    day, month, year, hour = get_time(tz)
    base_path = "./output"

    dir_path = Path(base_path) / year / month / day / hour
    dir_path.mkdir(parents=True, exist_ok=True)

    images_dir = dir_path / "images"
    images_dir.mkdir(exist_ok=True)
    output_path = Path("AI") / year / month / day / hour / "images"

    return images_dir, output_path


def rtsp_stream_init(rtsp_url):
    """Initialize RTSP stream with hardware acceleration"""
    video_cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # Optimize buffer settings
    video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    video_cap.set(cv2.CAP_PROP_FPS, 15)
    video_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
    video_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)

    # Hardware acceleration settings
    video_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

    try:
        video_cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    except Exception as e:
        pass

    try:
        video_cap.set(cv2.CAP_PROP_HW_DEVICE, 0)
    except:
        pass

    video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return video_cap\


def blur_face(image, frame_count):
    """Detect and blur faces in the image"""
    face_detector = YOLO("./models/face_bounding_rknn_model")
    results = face_detector.predict(image, conf=0.6, verbose=False)
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                face_region = image[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                image[y1:y2, x1:x2] = blurred_face
                logger.info(f"Blurred face at frame {frame_count}")
                print(f"Blurred face at frame {frame_count}")
    return image


def draw_tracking_results(frame, detections):
    """Draw bounding boxes on frame"""
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
    WHITE = (255, 255, 255)

    for det in detections:
        track_id = det['id']
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox

        color = [int(c) for c in COLORS[track_id % len(COLORS)]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id}", (x1 + 5, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    return frame


def main(args, listener):
    # Load configuration
    config = load_model_config()

    # Get API configuration
    api_config = config.get("api", {})
    post_endpoint = api_config.get("post_endpoint", "http://18.167.218.143:18000/pic/AIEarlyWarning")

    # Get device configuration
    device_config = config.get("device", {})
    device_id = device_config.get("device_id", "8")
    vin = device_config.get("vin", "as00086")
    picture_type = device_config.get("picture_type", 19)

    # Create output directories
    hong_kong_tz = pytz.timezone('Asia/Hong_Kong')

    # Get configuration from arguments
    model_type = args.type
    stream_url = args.stream
    blur_enabled = args.blur
    check_path = args.check_path

    video_cap = rtsp_stream_init(stream_url)
    logger.info(f"Starting RTSP stream processing: {stream_url}")
    print(f"Starting RTSP stream processing: {stream_url}")

    # Load model
    model_path = get_config("models", model_type, config)
    detector = YOLO(model_path)

    frame_count = 0
    PEOPLE_CAP = 2
    valid_path = ["a", "b", ""]
    frame_skip = 5

    # Track ID management
    track_counts = {}
    posted_track_ids = set()
    track_threshold = 5

    try:
        while True:
            path_name = listener.PathName
            success, frame = video_cap.read()
            if success:
                frame_count += 1

                # Skip frames
                if frame_count % frame_skip != 0:
                    continue

                # Check path validation if enabled
                if check_path and path_name not in valid_path:
                    logger.debug(f"Current path {path_name} not in detection area, skipping ....")
                    continue

                logger.info("Processing frame...")
                print("processing...")

                # Check and reset if needed
                check_and_reset_ids()

                images_dir, output_dir = create_output_directories(hong_kong_tz)

                if model_type in ["bicycle", "pets", "vehicle", "person"]:
                    class_list = get_config("classes", model_type, config)
                else:
                    class_list = [0]

                # Get detections
                logger.info(f"Start detection of {model_type}")
                print(f"Start detection of {model_type}: ")
                detections, bbox_list, track_ids_to_post = detection(
                    detector, frame,
                    get_config("confidence", model_type, config),
                    class_list, track_counts, posted_track_ids, track_threshold
                )

                # Filter track IDs that can be uploaded (24-hour check)
                valid_track_ids = [tid for tid in track_ids_to_post if can_upload_id(tid)]
                valid_detections = [det for det in detections if det['id'] in valid_track_ids]

                # Only proceed if there are valid track IDs to post
                if valid_detections and len(bbox_list) != 0:
                    # Apply blur if enabled
                    processed_img = frame.copy()
                    if blur_enabled:
                        logger.info(f"Start detection of faces")
                        print(f"Start detection of faces: ")
                        processed_img = blur_face(frame, frame_count)

                    # Draw bounding boxes
                    frame_with_boxes = draw_tracking_results(processed_img.copy(), valid_detections)

                    # Send to server with new endpoint structure
                    logger.info(f"Sending {len(valid_detections)} detections to server")
                    print(f"Sending {len(valid_detections)} detections to server")
                    send_detection_to_server(
                        frame_with_boxes,
                        valid_detections,
                        post_endpoint,
                        device_id,
                        vin,
                        picture_type
                    )

                    # Mark track IDs as posted
                    posted_track_ids.update(valid_track_ids)

                    logger.info(f"Frame {frame_count}: Posted {len(valid_track_ids)} new tracks with {len(bbox_list)} detections")
                    print(f"Frame {frame_count}: Posted {len(valid_track_ids)} new tracks with {len(bbox_list)} detections")
                elif len(bbox_list) != 0:
                    logger.info(f"Frame {frame_count}: {len(bbox_list)} detections (waiting for threshold or 24h reset)")
                    print(f"Frame {frame_count}: {len(bbox_list)} detections (waiting for threshold)")

            else:
                logger.error("Failed to read frame from RTSP stream")
                print("Failed to read frame from RTSP stream")
                video_cap.release()
                time.sleep(100)
                video_cap = rtsp_stream_init(stream_url)
                if not video_cap.isOpened():
                    logger.error("Failed to reconnect to RTSP stream")
                    print("Failed to reconnect to RTSP stream")
                    break

    except KeyboardInterrupt:
        logger.info("\nStopping RTSP stream processing...")
        print("\nStopping RTSP stream processing...")
    finally:
        logger.info(f"Total frames processed: {frame_count}")
        print(f"Total frames processed: {frame_count}")
        video_cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO Detection with New Endpoint Structure')
    parser.add_argument('--type', help='Model type/name (e.g., violence, license_plate, fire-and-smoke)', default="person", required=False)
    parser.add_argument('--stream', help='RTSP stream URL', default="rtsp://admin:rsxx1111@192.168.9.30", required=False)
    parser.add_argument('--blur', action='store_true', help='Enable blur on detected objects', default=True, required=False)
    parser.add_argument('--check-path', action='store_true', help='Enable path validation check', default=False, required=False)
    args = parser.parse_args()
    listener = TCP_Listener()
    listener.start()
    main(args, listener)
