import cv2
import numpy as np

def create_stereo_matcher():
    """StereoSGBM matcher; parameters resemble the "good calibration" setup
    in advanced_calibration_starter.py (P1/P2 penalties scaled to block_size)."""
    block_size = 9
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 13,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=1,
        speckleWindowSize=50,
        speckleRange=1,
        preFilterCap=40,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

def compute_disparity_map(left, right):
    stereo = create_stereo_matcher()
    disparity_map = stereo.compute(left, right).astype(np.float32) / 16.0
    return disparity_map

def compute_depth_map(disparity_map, fx, baseline):
    depth_map = np.zeros(disparity_map.shape, dtype=np.float32)
    valid = disparity_map > 0
    depth_map[valid] = (fx * baseline) / disparity_map[valid]
    return depth_map

def get_depth_and_disparity(left, right, fx, baseline):
    left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    disparity_map = compute_disparity_map(left_gray, right_gray)
    depth_map = compute_depth_map(disparity_map, fx, baseline)

    return depth_map, disparity_map

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_loading import DataLoader
    from visualization import depth_to_color

    dl = DataLoader("../kitty_data/drive1/2011_09_26_drive_0001_sync")
    left, right = dl.load_stereo_pair(0)
    depth_map, disparity_map = get_depth_and_disparity(left, right, dl.fx, dl.baseline)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].imshow(left)
    axes[0].set_title("Left image")
    axes[0].axis("off")

    im1 = axes[1].imshow(disparity_map, cmap="plasma")
    axes[1].set_title("Disparity map")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], label="disparity (px)")

    axes[2].imshow(depth_to_color(depth_map, vmax=80))
    axes[2].set_title("Depth map (m, colorized 0-80m)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
