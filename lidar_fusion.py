# point cloud to stereo
import numpy as np

def cart2hom(pts_3d):
    """ Input: nx3 points in Cartesian
        Output: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

def project_velo_to_image(pts_3d_velo, P, R0, V2C):
    '''
    Input: 3D points in Velodyne Frame [nx3]
    Output: 2D Pixels in Image Frame + camera-frame depth [nx3] -- (u, v, z_cam).
    z_cam (column 2) is the depth along the camera's own optical axis.
    '''
    R0_homo = np.vstack((R0, [0, 0, 0])) # Add Row of 0s (to be in homogeneous)
    R0_homo_2 = np.hstack((R0_homo, [[0], [0], [0], [1]]))# Add column of 0s (to be in homogeneous)
    p_r0 = P @ R0_homo_2 # P * R0 :                                           [3x4] @ [4x4] -> [3x4]
    p_r0_rt = p_r0 @ np.vstack((V2C, [0, 0, 0, 1])) # R*RO*R|T :               [3x4] @ [4x4] -> [3x4]

    pts_3d_homo = cart2hom(pts_3d_velo) # Add columns 0s(to be in homogeneous) [nx3] -> [nx4]
    p_r0_rt_x = p_r0_rt @ pts_3d_homo.T # P*R0*R|T*X (X being pts_3D)          [3x4] @ [4xn] -> [3xn]
    pts_2d = np.transpose(p_r0_rt_x)

    pts_2d[:, 0] = pts_2d[:, 0] / pts_2d[:, 2] #Convert Back to Cartesian
    pts_2d[:, 1] = pts_2d[:, 1] / pts_2d[:, 2] #Convert Back to Cartesian
    return pts_2d[:, 0:3]

def get_lidar_in_image_fov(pc_velo, P, R0, V2C, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = project_velo_to_image(pc_velo, P, R0, V2C)  # Project Velodyne to Camera Image
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance) # We don't want things that are closer to the clip distance (2m)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo

def build_lidar_depth_map(pts_2d, depths, image_shape):
    """Rasterize (pixel, depth) pairs into a sparse (H, W) depth map,
    0 = no LiDAR return at that pixel -- same convention as compute_depth_map() in stereo_2_depth.py, so the two are directly comparable.

    Where multiple points round to the same pixel, keep the nearest depth.
    """
    h, w = image_shape
    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    # Rounding can push an in-FOV point -- clip to stay in bounds.
    us = np.clip(np.round(pts_2d[:, 0]).astype(int), 0, w - 1)
    vs = np.clip(np.round(pts_2d[:, 1]).astype(int), 0, h - 1)
    np.minimum.at(depth_map, (vs, us), depths)
    depth_map[np.isinf(depth_map)] = 0
    return depth_map

def get_lidar_depth_and_maps(pc_velo, image_shape, P, R0, V2C):
    """Given a LiDAR point cloud and calibration, project into the image and return both the per-point (pixel, depth) data
     and the rasterized (H, W) depth maps -- for both the raw (LiDAR-frame) and corrected (camera-frame) depth definitions.
    """
    h, w = image_shape
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, P, R0, V2C, 0, 0, w, h, return_more=True
    )
    pts_2d_fov = pts_2d[fov_inds]

    raw_depths = imgfov_pc_velo[:, 0]  # LiDAR-frame forward axis (reference's shortcut)
    cam_depths = pts_2d_fov[:, 2]      # camera-frame Z (ours)

    raw_map = build_lidar_depth_map(pts_2d_fov[:, :2], raw_depths, (h, w))
    cam_map = build_lidar_depth_map(pts_2d_fov[:, :2], cam_depths, (h, w))

    return pts_2d_fov, raw_depths, cam_depths, raw_map, cam_map

if __name__ == "__main__":
    from data_loading import DataLoader

    dl = DataLoader("../kitty_data/drive1/2011_09_26_drive_0001_sync")
    left, right = dl.load_stereo_pair(0)
    pc_velo = dl.load_point_cloud(0)
    h, w = left.shape[:2]

    pts_2d_fov, raw_depths, cam_depths, raw_map, cam_map = get_lidar_depth_and_maps(
        pc_velo, (h, w), dl.P, dl.R0, dl.V2C
    )

    print("nonzero pixels: raw", (raw_map > 0).sum(), "cam", (cam_map > 0).sum())

    diff = cam_depths - raw_depths
    print("depth diff (cam - raw): mean=%.4f  std=%.4f  min=%.4f  max=%.4f" % (
        diff.mean(), diff.std(), diff.min(), diff.max()
    ))

    near = raw_depths < 5
    far = raw_depths > 40
    print("mean diff for near (<5m) points:", diff[near].mean(), "n=", near.sum())
    print("mean diff for far (>40m) points:", diff[far].mean(), "n=", far.sum())

    # Sanity-check visualization: (colored by cam_depths)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(left)
    sc = ax.scatter(pts_2d_fov[:, 0], pts_2d_fov[:, 1], c=cam_depths, cmap="viridis_r", vmax=80, s=4)
    ax.set_title("LiDAR points projected onto left image (camera-frame depth)")
    ax.axis("off")
    fig.colorbar(sc, ax=ax, label="depth (m)")
    plt.tight_layout()
    plt.show()

