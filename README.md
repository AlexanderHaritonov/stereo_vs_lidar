# stereo_vs_lidar

Estimate depth from stereo vision and compare against LiDAR point cloud.
Run on frame sequences from KITTY Raw datasets and produce a video(s) visualizing the two side by side.

## Setup

```
pip install -r requirements.txt
wget https://stereo-vision.s3.eu-west-3.amazonaws.com/yolo11n.pt
```

Run both from inside this folder (`stereo_vs_lidar/`) -- `detection.py` loads the model via the relative path `yolo11n.pt`.
