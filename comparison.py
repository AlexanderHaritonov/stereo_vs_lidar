import numpy as np

def compare_depth_maps(stereo_depth_map, lidar_depth_map):
    """MAE/RMSE between stereo and LiDAR depth maps, restricted to pixels where both have a valid (nonzero) reading.
    Assumes the two maps are already the same shape and share the 0=invalid convention."""
    valid = (stereo_depth_map > 0) & (lidar_depth_map > 0)
    n = int(valid.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    diff = stereo_depth_map[valid] - lidar_depth_map[valid]
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())
    return mae, rmse, n

def compare_depth_maps_in_box(stereo_depth_map, lidar_depth_map, box):
    """compare_depth_maps() restricted to a pixel box (x1, y1, x2, y2),
    plus the per-map nearest distance for the label.
    Returns None when the box has no pixel with both readings."""
    x1, y1, x2, y2 = box
    stereo_crop = stereo_depth_map[y1:y2, x1:x2]
    lidar_crop = lidar_depth_map[y1:y2, x1:x2]

    mae, rmse, n = compare_depth_maps(stereo_crop, lidar_crop)
    if n == 0:
        return None

    return {
        "stereo_nearest": stereo_crop[stereo_crop > 0].min(),
        "lidar_nearest": lidar_crop[lidar_crop > 0].min(),
        "mae": mae,
        "rmse": rmse,
        "n": n,
    }

def format_box_label(stats):
    """Single-line label for a detection box: nearest stereo/lidar depth (or a no-data note)."""
    if stats is None:
        return "no lidar data"
    return f"{stats['stereo_nearest']:.1f} / {stats['lidar_nearest']:.1f}m"

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_loading import DataLoader
    from stereo_2_depth import get_depth_and_disparity
    from lidar_fusion import get_lidar_depth_and_maps
    from detection import load_model, run_obstacle_detection
    from visualization import overlay_points_in_boxes_on_image, draw_boxes_with_labels

    dl = DataLoader("../kitty_data/drive1/2011_09_26_drive_0001_sync")
    left, right = dl.load_stereo_pair(0)
    pc_velo = dl.load_point_cloud(0)
    h, w = left.shape[:2]

    stereo_depth_map, _ = get_depth_and_disparity(left, right, dl.fx, dl.baseline)
    pts_2d_fov, _, cam_depths, _, lidar_depth_map = get_lidar_depth_and_maps(pc_velo, (h, w), dl.P, dl.R0, dl.V2C)

    mae, rmse, n = compare_depth_maps(stereo_depth_map, lidar_depth_map)
    print(f"compared {n} pixels with both stereo and lidar returns")
    print(f"MAE:  {mae:.3f} m")
    print(f"RMSE: {rmse:.3f} m")

    model = load_model()
    boxes, _, _ = run_obstacle_detection(model, left)

    texts = []
    for box in boxes:
        stats = compare_depth_maps_in_box(stereo_depth_map, lidar_depth_map, box)
        texts.append(format_box_label(stats))
        print(box, texts[-1])

    vis = overlay_points_in_boxes_on_image(left, pts_2d_fov[:, :2], cam_depths, boxes, vmax=80)
    vis = draw_boxes_with_labels(vis, boxes, texts)

    plt.figure(figsize=(14, 5))
    plt.imshow(vis)
    plt.title("Detections with stereo/lidar median depth + lidar points in box")
    plt.axis("off")
    plt.show()
