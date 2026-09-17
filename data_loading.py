import os

import cv2
import numpy as np

class DataLoader:
    """Loads calib and stereo image pairs from a KITTI sync folder."""

    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.fx, self.baseline = self._get_fx_and_baseline()

    def _get_fx_and_baseline(self):
        """Focal length (fx) and stereo baseline (m) for the image_02/image_03 pair."""
        cam_to_cam_calib_file = os.path.join(os.path.dirname(self.root_folder), "calib", "calib_cam_to_cam.txt")
        calib = read_calib_file(cam_to_cam_calib_file)
        P_rect_02 = calib["P_rect_02"].reshape(3, 4)
        P_rect_03 = calib["P_rect_03"].reshape(3, 4)
        fx = P_rect_02[0, 0]
        baseline = (P_rect_02[0, 3] - P_rect_03[0, 3]) / fx
        return fx, baseline

    def load_stereo_pair(self, frame_number):
        """Load the matching left (image_02) / right (image_03) frame from the sync folder."""
        filename = f"{frame_number:010d}.png"
        left_path = os.path.join(self.root_folder, "image_02", "data", filename)
        right_path = os.path.join(self.root_folder, "image_03", "data", filename)
        return load_image(left_path), load_image(right_path)

def load_image(path):
    """Read an image file and return it as an RGB numpy array (H, W, 3)."""
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # channel swap

def read_calib_file(cam_to_cam_calib_file):
    """ Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(cam_to_cam_calib_file, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def get_q_matrix(cam_to_cam_calib_file):
    """Disparity-to-depth mapping matrix (Q), derived via cv2.stereoRectify from the raw (unrectified) intrinsics/extrinsics of cam2 and cam3
     - independent of the P_rect-based fx/baseline shortcut in get_fx_and_baseline()."""
    calib = read_calib_file(cam_to_cam_calib_file)
    K_02 = calib["K_02"].reshape(3, 3)
    D_02 = calib["D_02"]
    K_03 = calib["K_03"].reshape(3, 3)
    D_03 = calib["D_03"]
    R_02 = calib["R_02"].reshape(3, 3)
    T_02 = calib["T_02"].reshape(3, 1)
    R_03 = calib["R_03"].reshape(3, 3)
    T_03 = calib["T_03"].reshape(3, 1)
    image_size = (int(calib["S_02"][0]), int(calib["S_02"][1]))

    # R_0i/T_0i are each camera's pose relative to cam0 -- compose them
    # to get the relative pose from cam2 to cam3, which is what stereoRectify needs.
    R_23 = R_03 @ R_02.T
    T_23 = T_03 - R_23 @ T_02

    _, _, _, _, Q, _, _ = cv2.stereoRectify(K_02, D_02, K_03, D_03, image_size, R_23, T_23)
    return Q
