import os

import cv2

from data_loading import DataLoader
from stereo_2_depth import get_depth_and_disparity
from lidar_fusion import get_lidar_depth_and_maps
from detection import load_model, run_obstacle_detection
from comparison import compare_depth_maps, compare_depth_maps_in_box, format_box_label
from visualization import depth_to_color, overlay_points_in_boxes_on_image, draw_boxes_with_labels

def _count_frames(root_folder):
    """Upper bound for the frame loop: the highest frame index present in any of image_02/image_03/velodyne_points."""
    max_index = -1
    for subdir, ext in (("image_02", ".png"), ("image_03", ".png"), ("velodyne_points", ".bin")):
        data_dir = os.path.join(root_folder, subdir, "data")
        for f in os.listdir(data_dir):
            if f.endswith(ext):
                max_index = max(max_index, int(f[:10]))
    return max_index + 1

def _build_frame(dl, model, frame_number, vmax):
    left, right = dl.load_stereo_pair(frame_number)
    pc_velo = dl.load_point_cloud(frame_number)
    h, w = left.shape[:2]

    stereo_depth_map, _ = get_depth_and_disparity(left, right, dl.fx, dl.baseline)
    pts_2d_fov, _, cam_depths, _, lidar_depth_map = get_lidar_depth_and_maps(pc_velo, (h, w), dl.P, dl.R0, dl.V2C)

    boxes, _, _ = run_obstacle_detection(model, left)
    texts = [format_box_label(compare_depth_maps_in_box(stereo_depth_map, lidar_depth_map, box)) for box in boxes]

    camera_panel = overlay_points_in_boxes_on_image(left, pts_2d_fov[:, :2], cam_depths, boxes, vmax)
    camera_panel = draw_boxes_with_labels(camera_panel, boxes, texts)

    mae, rmse, _ = compare_depth_maps(stereo_depth_map, lidar_depth_map)
    cv2.putText(camera_panel, f"MAE: {mae:.2f}m", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 40, 30), 2)
    cv2.putText(camera_panel, f"RMSE: {rmse:.2f}m", (10, 55), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 40, 30), 2)

    stereo_panel = depth_to_color(stereo_depth_map, vmax)
    lidar_panel = depth_to_color(lidar_depth_map, vmax)

    half_h = h // 2
    side_w = w // 2
    lidar_panel = cv2.resize(lidar_panel, (side_w, half_h))
    stereo_panel = cv2.resize(stereo_panel, (side_w, h - half_h))
    side_col = cv2.vconcat([lidar_panel, stereo_panel])
    return cv2.hconcat([camera_panel, side_col])

def make_comparison_video(root_folder, output_dir="output", fps=10, vmax=80):
    """Render the camera+boxes/stereo/lidar 3-panel comparison for every frame in root_folder into output_dir/<sequence name>.mp4."""
    dl = DataLoader(root_folder)
    model = load_model()
    frames_cnt = _count_frames(root_folder)

    result_video = []
    for idx in range(frames_cnt):
        print(idx + 1, "of", frames_cnt)
        try:
            result_video.append(_build_frame(dl, model, idx, vmax))
        except (FileNotFoundError, cv2.error):
            print(f"skipping frame {idx}: missing image or lidar file")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(root_folder.rstrip("/")) + ".mp4")

    h, w = result_video[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (w, h))
    print(out.isOpened())  # sanity check

    for frame in result_video:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    return output_path

if __name__ == "__main__":
    output_path = make_comparison_video("../kitty_data/drive14/2011_09_26_drive_0014_sync")
    print(f"wrote {output_path}")
