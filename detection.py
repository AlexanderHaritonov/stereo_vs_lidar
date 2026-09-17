import cv2
from ultralytics import YOLO

def load_model(weights_path="yolo11n.pt"):
    return YOLO(weights_path)

def run_obstacle_detection(model, image, classes=("person", "car")):
    """Detect objects in image, filtered to classes. Returns boxes as pixel (x1, y1, x2, y2) ints, plus matching labels and confidences."""
    result = model.predict(image, verbose=False)[0]

    boxes, labels, confs = [], [], []
    for box in result.boxes:
        label = result.names[int(box.cls[0])]
        if label not in classes:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
        labels.append(label)
        confs.append(float(box.conf[0]))

    return boxes, labels, confs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_loading import DataLoader

    dl = DataLoader("../kitty_data/drive1/2011_09_26_drive_0001_sync")
    left, _ = dl.load_stereo_pair(0)

    model = load_model()
    boxes, labels, confs = run_obstacle_detection(model, left)
    print(f"detected {len(boxes)} boxes: {list(zip(labels, [round(c, 2) for c in confs]))}")

    vis = left.copy()
    for (x1, y1, x2, y2), label, conf in zip(boxes, labels, confs):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 40, 30), 2)
        cv2.putText(vis, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 40, 30), 2)

    plt.figure(figsize=(14, 5))
    plt.imshow(vis)
    plt.title("YOLO detections (person/car)")
    plt.axis("off")
    plt.show()
