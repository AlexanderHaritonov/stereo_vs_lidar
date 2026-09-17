import cv2

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
