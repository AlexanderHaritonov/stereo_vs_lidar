# stereo_vs_lidar

Estimate depth from stereo vision and compare against LiDAR point cloud.
Run on frame sequences from KITTY Raw datasets and produce a video(s) visualizing the two side by side.
Mean Absolute Error and Root Mean Squared Error between LiDAR and each of two depth estimates — stereo (SGBM) and monocular (Depth Anything V2).
Also, per detection box, nearest distance as per mono / stereo / lidar.

## Models

- **YOLO11n** (`yolo11n.pt`) — object detection (person/car boxes)
- **Depth Anything V2** (`depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`) — monocular depth estimation

## Setup

```
pip install -r requirements.txt
wget https://stereo-vision.s3.eu-west-3.amazonaws.com/yolo11n.pt
```

Run both from inside this folder (`stereo_vs_lidar/`) -- `detection.py` loads the model via the relative path `yolo11n.pt`.

## Demo

![demo](output/preview.gif)

[Full video](output/preview.mp4)
