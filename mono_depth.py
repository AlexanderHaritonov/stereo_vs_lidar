import cv2
from PIL import Image
from transformers import pipeline

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

def load_mono_depth_model():
    return pipeline("depth-estimation", model=MODEL_ID, device="cpu")

def get_mono_depth(image, model):
    """Monocular metric depth map (m) for a single RGB image, resized to match it."""
    h, w = image.shape[:2]
    result = model(Image.fromarray(image))
    depth_map = result["predicted_depth"].numpy()
    if depth_map.shape != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth_map

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_loading import DataLoader
    from visualization import depth_to_color

    dl = DataLoader("../kitty_data/drive1/2011_09_26_drive_0001_sync")
    left, _ = dl.load_stereo_pair(0)

    model = load_mono_depth_model()
    mono_depth_map = get_mono_depth(left, model)

    print(f"mono depth: min {mono_depth_map.min():.2f}m, max {mono_depth_map.max():.2f}m")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].imshow(left)
    axes[0].set_title("Left image")
    axes[0].axis("off")

    axes[1].imshow(depth_to_color(mono_depth_map, vmax=80))
    axes[1].set_title("Mono depth (Depth Anything V2, m, colorized 0-80m)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
