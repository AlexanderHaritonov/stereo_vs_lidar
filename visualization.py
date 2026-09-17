import cv2
import numpy as np
from matplotlib import colormaps

# HSV lookup table point circles (plt.cm.get_cmap("hsv", 256));
_HSV_LUT = (np.array([colormaps["hsv"](i) for i in range(256)])[:, :3] * 255).astype(np.uint8)

def depth_to_color(depth_map, vmax):
    """Colorize a dense depth map:
    near = high index (red end), far = low index (violet end),
    normalized by vmax and vectorized over the whole map."""
    normalized = np.clip(depth_map, 0, vmax) / vmax
    idx = ((1.0 - normalized) * 255).astype(np.uint8)
    color = _HSV_LUT[idx].copy()
    color[depth_map <= 0] = 0  # invalid (no depth) -> blank
    return color

def overlay_points_on_image(image, pts_2d, depths, vmax):
    """Draw a filled circle at each 2D point, colored by depth."""
    out = image.copy()
    normalized = np.clip(depths, 0, vmax) / vmax
    idx = ((1.0 - normalized) * 255).astype(np.uint8)
    colors = _HSV_LUT[idx]
    for (x, y), color in zip(pts_2d, colors):
        cv2.circle(out, (int(np.round(x)), int(np.round(y))), 2,
                   color=tuple(int(c) for c in color), thickness=-1)
    return out

def points_in_boxes_mask(pts_2d, boxes, shrink_factor=0.2):
    """Mask over pts_2d, True where the point falls inside any box (x1, y1, x2, y2)."""
    mask = np.zeros(len(pts_2d), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        if shrink_factor:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half_w = (x2 - x1) / 2 * (1 - shrink_factor)
            half_h = (y2 - y1) / 2 * (1 - shrink_factor)
            x1, x2 = cx - half_w, cx + half_w
            y1, y2 = cy - half_h, cy + half_h
        mask |= (pts_2d[:, 0] > x1) & (pts_2d[:, 0] < x2) & (pts_2d[:, 1] > y1) & (pts_2d[:, 1] < y2)
    return mask

def overlay_points_in_boxes_on_image(image, pts_2d, depths, boxes, vmax, shrink_factor=0.2):
    """overlay_points_on_image(), restricted to points inside one of boxes."""
    mask = points_in_boxes_mask(pts_2d, boxes, shrink_factor)
    return overlay_points_on_image(image, pts_2d[mask], depths[mask], vmax)

def draw_boxes_with_labels(image, boxes, texts, color=(255, 40, 30)):
    """Rectangle + label above it, for each box."""
    out = image.copy()
    for (x1, y1, x2, y2), text in zip(boxes, texts):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, text, (x1, y1 - 4), cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)
    return out
