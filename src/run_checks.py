import sys
sys.path.append("../checks")

import cv2
import pandas as pd

from loader import loader
from image_quality import blur, brightness, resolution, contrast
from label_quality import out_of_bounds, zero_area, invalid_class

def run_checks(records, run_type="clean"):
    row = []
    for record in records:
        img = cv2.imread(str(record["image_path"]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur_rec = blur(gray)
        bright_rec = brightness(gray)
        contr_rec = contrast(gray)
        res_rec = resolution(img)

        oob_rec = out_of_bounds(record["boxes"])
        zero_rec = zero_area(record["boxes"])
        cls_rec = invalid_class(record["boxes"])

        row.append({
            "frame_id": record["frame_id"],
            "image_uri": str(record["image_path"]),
            "run_type": run_type,
            "blur_score": blur_rec["metric"],
            "brightness": bright_rec["metric"],
            "contrast": contr_rec["metric"],
            "oob_count": oob_rec["metric"],
            "zero_count": zero_rec["metric"],
            "invalid_class": cls_rec["metric"],
            "resolution": res_rec["metric"]
        })

    return pd.DataFrame(row)


if __name__ == "__main__":
    train_rec = loader("../datasets/coco8/images/train", "../datasets/coco8/labels/train")
    val_rec   = loader("../datasets/coco8/images/val",   "../datasets/coco8/labels/val")
    records = train_rec + val_rec

    df = run_checks(records)

    print(df)
    print(df.shape)
    print(df.describe())
