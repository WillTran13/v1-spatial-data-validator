"""
Run the simulator. Then measure both the clean and degraded sets
"""
import cv2
import sys

sys.path.append("../checks")

import pandas as pd

from drift_simulator import DriftSimulator
from loader import loader
from run_checks import run_checks

DIR_PAIRS = [
    ("../datasets/coco8/images/train", "../datasets/coco8/labels/train"),
    ("../datasets/coco8/images/val", "../datasets/coco8/labels/val"),
]
DEGRADED_DIR = "../data/degraded"

def main():
    sim = DriftSimulator(DIR_PAIRS, DEGRADED_DIR)
    dimension_rows = sim.run()
    dims = pd.DataFrame(dimension_rows)

    degraded_records = loader(f"{DEGRADED_DIR}/images", f"{DEGRADED_DIR}/labels")
    degraded_df = run_checks(degraded_records, run_type="degraded")

    clean_records = []
    for img_dir, lbl_dir in DIR_PAIRS:
        clean_records.extend(loader(img_dir, lbl_dir))
    clean_df = run_checks(clean_records, run_type="clean")

    degraded_df = degraded_df.merge(dims, on="frame_id", how="left")
    clean_df = clean_df.merge(dims, on="frame_id", how="left")

    clean_df["drift_severity"] = 0.0

    combined = pd.concat([clean_df, degraded_df], ignore_index=True)
    combined = combined.sort_values(
        ["sensor_id", "captured_timestamp", "run_type"]
    ).reset_index(drop=True)

    assert not combined.isnull().any().any(), "nulls after merge"
    print("rows:", len(combined), " columns:", len(combined.columns))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(combined[[
        "frame_id", "sensor_id", "captured_timestamp", "run_type",
        "drift_severity", "blur_score", "brightness",
    ]])

    return combined

if __name__ == "__main__":
    main()