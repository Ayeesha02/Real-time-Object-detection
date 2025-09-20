# Real-Time Object Detection with YOLOv3

A Python application for real-time object detection using YOLOv3 and OpenCV. This project provides an efficient implementation for detecting objects in video streams from webcams or video files.

## Features

- Real-time object detection using YOLOv3
- Support for webcam input
- Multiple object class detection (80 COCO classes)
- Configurable confidence and NMS thresholds
- FPS counter and performance metrics
- GPU acceleration support (when CUDA is available)
- Frame capture functionality
- Debug mode for detailed logging

## Prerequisites

- Python 3.8+
- OpenCV (opencv-python)
- NumPy
- urllib3

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Project_detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install opencv-python numpy urllib3
   ```

4. Download YOLOv3 weights:
   ```bash
   python download_yolov3.py
   ```
   If the automatic download fails, manually download the weights from:
   - YOLOv3 weights: [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
   Place the downloaded file in the project directory.

## Usage

Run the object detection:
```bash
python main.py [options]
```

### Command Line Options

- `--camera`: Camera device index (default: 0)
- `--width`: Frame width (default: 640)
- `--height`: Frame height (default: 480)
- `--confidence`: Confidence threshold (default: 0.3)
- `--nms`: Non-maximum suppression threshold (default: 0.4)
- `--debug`: Enable debug mode

### Controls

While the application is running:
- Press 'q' to quit
- Press 's' to save the current frame

## Project Structure

- `main.py`: Main application script with YOLOv3 implementation
- `download_yolov3.py`: Helper script for downloading model weights
- `yolov3.cfg`: YOLOv3 model configuration
- `coco.names`: COCO dataset class names
- `yolov3.weights`: Pre-trained model weights (downloaded separately)

## Configuration

The application includes several configurable parameters in `main.py`:

- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections (default: 0.3)
- `NMS_THRESHOLD`: Non-maximum suppression threshold (default: 0.4)
- `INPUT_WIDTH/HEIGHT`: Input resolution for the neural network (default: 608x608)

## Performance Notes

- Higher input resolution (608x608) provides better accuracy but slower performance
- GPU acceleration is automatically used if CUDA is available
- Adjust confidence and NMS thresholds to balance between accuracy and false positives

## Troubleshooting

### Common Issues

1. Camera Access on macOS:
   - Go to System Settings > Privacy & Security > Camera
   - Enable camera access for Terminal or VS Code

2. YOLOv3 Weights Download:
   - If automatic download fails, use the manual download link
   - Verify the file size is approximately 248MB

3. OpenCV Installation:
   - If you encounter issues, try: `pip install --upgrade opencv-python`

### Debug Mode

Run with debug mode for detailed logging:
```bash
python main.py --debug
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv3 by Joseph Redmon (https://pjreddie.com/darknet/yolo/)
- OpenCV team for their computer vision library