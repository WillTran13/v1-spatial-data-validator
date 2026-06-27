import cv2
from pathlib import Path
from loader import loader


def draw_boxes(record, output_dir="outputs/reports"):
    img = cv2.imread(str(record["image_path"]))
    img_h, img_w = img.shape[:2]   # note: shape is (height, width, channels)

    for box in record["boxes"]:
        cls, xc, yc, w, h = box
        # YOLO normalized -> pixel corners (the Part 3 conversion)
        px_c, py_c = xc * img_w, yc * img_h
        pw, ph = w * img_w, h * img_h
        x1, y1 = int(px_c - pw / 2), int(py_c - ph / 2)
        x2, y2 = int(px_c + pw / 2), int(py_c + ph / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(cls), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = Path(output_dir) / f"{record['frame_id']}_boxed.jpg"
    cv2.imwrite(str(out_path), img)

if __name__ == "__main__":
    train_records = loader("../datasets/coco8/images/train", "../datasets/coco8/labels/train")
    val_records = loader("../datasets/coco8/images/val", "../datasets/coco8/labels/val")

    records = train_records + val_records

    for record in records:
        draw_boxes(record,"../outputs/reports")