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
    plus the per-map median distance for the label.
    Returns None when the box has no pixel with both readings."""
    x1, y1, x2, y2 = box
    stereo_crop = stereo_depth_map[y1:y2, x1:x2]
    lidar_crop = lidar_depth_map[y1:y2, x1:x2]

    mae, rmse, n = compare_depth_maps(stereo_crop, lidar_crop)
    if n == 0:
        return None

    return {
        "stereo_median": np.median(stereo_crop[stereo_crop > 0]),
        "lidar_median": np.median(lidar_crop[lidar_crop > 0]),
        "mae": mae,
        "rmse": rmse,
        "n": n,
    }

if __name__ == "__main__":
    from data_loading import DataLoader
    from stereo_2_depth import get_depth_and_disparity
    from lidar_fusion import get_lidar_depth_and_maps

    dl = DataLoader("../kitty_data/drive1/2011_09_26_drive_0001_sync")
    left, right = dl.load_stereo_pair(0)
    pc_velo = dl.load_point_cloud(0)
    h, w = left.shape[:2]

    stereo_depth_map, _ = get_depth_and_disparity(left, right, dl.fx, dl.baseline)
    _, _, _, _, lidar_depth_map = get_lidar_depth_and_maps(pc_velo, (h, w), dl.P, dl.R0, dl.V2C)

    mae, rmse, n = compare_depth_maps(stereo_depth_map, lidar_depth_map)
    print(f"compared {n} pixels with both stereo and lidar returns")
    print(f"MAE:  {mae:.3f} m")
    print(f"RMSE: {rmse:.3f} m")

    # Per-box sanity check -- manual box for now, stands in for a YOLO detection later.
    test_box = (w // 4, h // 2, w // 2, h)
    box_stats = compare_depth_maps_in_box(stereo_depth_map, lidar_depth_map, test_box)
    if box_stats is None:
        print(f"box {test_box}: no lidar returns in range")
    else:
        print(f"box {test_box}: stereo={box_stats['stereo_median']:.2f}m "
              f"lidar={box_stats['lidar_median']:.2f}m "
              f"MAE={box_stats['mae']:.3f}m RMSE={box_stats['rmse']:.3f}m n={box_stats['n']}")
