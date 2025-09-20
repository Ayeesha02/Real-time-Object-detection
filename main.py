#!/usr/bin/env python3
"""
Real-time object detection using YOLOv3 with improved accuracy and reliability.
Fixed issues: proper YOLOv3 URLs, correct preprocessing, better error handling.
"""

import cv2
import numpy as np
import time
import argparse
import os
import sys
import urllib.request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Updated URLs for YOLOv3 (more accurate than YOLOv2)
MODEL_URL = "https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights"
CONFIG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# File names
MODEL_FILE = "yolov3.weights"
CONFIG_FILE = "yolov3.cfg"
NAMES_FILE = "coco.names"

# Detection parameters (optimized for better accuracy)
CONFIDENCE_THRESHOLD = 0.3  # Lowered for better detection
NMS_THRESHOLD = 0.4
INPUT_WIDTH = 608  # Higher resolution for better accuracy
INPUT_HEIGHT = 608

# Visualization parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_COLOR = (0, 255, 0)
LINE_TYPE = 2
BOX_THICKNESS = 2

# Color palette for different classes
COLORS = np.random.uniform(0, 255, size=(80, 3))  # COCO has 80 classes


def download_file(url, dest, chunk_size=8192):
    """Download file with progress tracking and error handling."""
    if os.path.exists(dest):
        # Check if file size is reasonable
        file_size = os.path.getsize(dest)
        if dest.endswith('.weights') and file_size < 50 * 1024 * 1024:  # Less than 50MB
            logger.warning(f"{dest} exists but seems too small ({file_size} bytes). Re-downloading...")
            os.remove(dest)
        else:
            logger.info(f"{dest} already exists. Skipping download.")
            return

    logger.info(f"Downloading {url} to {dest}...")
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = response.getheader('Content-Length')
            total_size = int(total_size) if total_size else None
            
            downloaded = 0
            with open(dest, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size:
                        percent = (downloaded * 100) // total_size
                        if downloaded % (chunk_size * 100) == 0:  # Log every ~800KB
                            logger.info(f"Downloaded {downloaded:,}/{total_size:,} bytes ({percent}%)")
                    else:
                        if downloaded % (chunk_size * 100) == 0:
                            logger.info(f"Downloaded {downloaded:,} bytes")
        
        logger.info(f"Download complete: {dest} ({downloaded:,} bytes)")
        
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error {e.code}: {e.reason}")
        if e.code == 403:
            logger.error("Download blocked. Please download manually:")
            logger.error(f"URL: {url}")
            logger.error(f"Save as: {dest}")
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def validate_weights_file(filepath):
    """Validate that the weights file is legitimate."""
    try:
        if not os.path.exists(filepath):
            return False, "File does not exist"
        
        file_size = os.path.getsize(filepath)
        if file_size < 200 * 1024 * 1024:  # YOLOv3 should be ~248MB
            return False, f"File too small ({file_size:,} bytes, expected ~248MB)"
        
        # Check if it starts with HTML (common when download fails)
        with open(filepath, 'rb') as f:
            header = f.read(100)
            if header.startswith(b'<') or b'<html' in header.lower():
                return False, "File appears to be HTML (download failed)"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {e}"


def load_yolo_model():
    """Download and load YOLO model files."""
    logger.info("Loading YOLO model...")
    
    # Download required files
    try:
        download_file(CONFIG_URL, CONFIG_FILE)
        download_file(NAMES_URL, NAMES_FILE)
        download_file(MODEL_URL, MODEL_FILE)
    except Exception as e:
        logger.error("Failed to download YOLO files.")
        logger.error("Manual download required:")
        logger.error(f"Config: {CONFIG_URL} -> {CONFIG_FILE}")
        logger.error(f"Names: {NAMES_URL} -> {NAMES_FILE}")
        logger.error(f"Weights: {MODEL_URL} -> {MODEL_FILE}")
        sys.exit(1)
    
    # Validate weights file
    is_valid, message = validate_weights_file(MODEL_FILE)
    if not is_valid:
        logger.error(f"Invalid weights file: {message}")
        logger.error(f"Please download manually: {MODEL_URL}")
        sys.exit(1)
    
    # Load the network
    try:
        logger.info("Loading neural network...")
        net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, MODEL_FILE)
        
        # Set computation backend
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Check if CUDA is available for GPU acceleration
        try:
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logger.info("Using GPU acceleration (CUDA)")
        except:
            logger.info("Using CPU computation")
        
        # Load class names
        with open(NAMES_FILE, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        logger.info(f"Model loaded successfully. Classes: {len(classes)}")
        return net, classes
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


def get_output_layers(net):
    """Get output layer names from the network."""
    layer_names = net.getLayerNames()
    try:
        unconnected = net.getUnconnectedOutLayers()
        # Handle different OpenCV versions
        if len(unconnected.shape) == 1:
            return [layer_names[i - 1] for i in unconnected]
        else:
            return [layer_names[i[0] - 1] for i in unconnected]
    except Exception as e:
        logger.error(f"Error getting output layers: {e}")
        return []


def draw_detection(img, class_id, confidence, x, y, w, h, classes):
    """Draw bounding box and label for detected object."""
    color = COLORS[class_id % len(COLORS)]
    label = f"{classes[class_id]}: {confidence:.2f}"
    
    # Draw bounding box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, BOX_THICKNESS)
    
    # Draw label background
    label_size, _ = cv2.getTextSize(label, FONT, FONT_SCALE, LINE_TYPE)
    cv2.rectangle(img, (x, y - label_size[1] - 10), 
                  (x + label_size[0], y), color, -1)
    
    # Draw label text
    cv2.putText(img, label, (x, y - 5), FONT, FONT_SCALE, 
                (255, 255, 255), LINE_TYPE)


def process_detections(outputs, frame_width, frame_height, classes):
    """Process YOLO outputs and return filtered detections."""
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            # Extract confidence scores for all classes
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONFIDENCE_THRESHOLD:
                # YOLO returns center coordinates and dimensions
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                
                # Calculate top-left coordinates
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                              CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    detections = []
    if len(indices) > 0:
        # Handle different OpenCV versions
        if isinstance(indices, np.ndarray) and len(indices.shape) == 2:
            indices = indices.flatten()
        
        for i in indices:
            detections.append({
                'box': boxes[i],
                'confidence': confidences[i],
                'class_id': class_ids[i],
                'class_name': classes[class_ids[i]]
            })
    
    return detections


def process_frame(frame, net, classes, output_layers):
    """Process a single frame for object detection."""
    height, width = frame.shape[:2]
    
    # Create blob from frame
    # YOLOv3 expects values normalized to [0,1] range
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), 
                                swapRB=True, crop=False)
    
    # Set input and run forward pass
    net.setInput(blob)
    start_time = time.time()
    outputs = net.forward(output_layers)
    inference_time = time.time() - start_time
    
    # Process detections
    detections = process_detections(outputs, width, height, classes)
    
    # Draw detections
    for detection in detections:
        box = detection['box']
        x, y, w, h = box
        draw_detection(frame, detection['class_id'], detection['confidence'], 
                      x, y, w, h, classes)
    
    # Add performance info
    fps_text = f"Inference: {inference_time*1000:.1f}ms | Detections: {len(detections)}"
    cv2.putText(frame, fps_text, (10, 30), FONT, 0.7, (0, 255, 255), 2)
    
    return frame, detections


def main():
    parser = argparse.ArgumentParser(description="Real-time YOLOv3 Object Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS threshold")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Update thresholds if provided
    global CONFIDENCE_THRESHOLD, NMS_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence
    NMS_THRESHOLD = args.nms
    
    # Load YOLO model
    net, classes = load_yolo_model()
    output_layers = get_output_layers(net)
    
    # Initialize camera
    logger.info(f"Initializing camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
    logger.info("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Process frame
            processed_frame, detections = process_frame(frame, net, classes, output_layers)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                logger.info(f"Average FPS: {fps:.1f}")
            
            # Show frame
            cv2.imshow("YOLOv3 Object Detection", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, processed_frame)
                logger.info(f"Frame saved as {filename}")
            
            # Print detections in debug mode
            if args.debug and detections:
                detection_info = ", ".join([f"{d['class_name']}({d['confidence']:.2f})" 
                                          for d in detections])
                logger.debug(f"Detected: {detection_info}")
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    main()