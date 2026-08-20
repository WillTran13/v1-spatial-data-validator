import cv2
from loader import loader
from datetime import datetime, timedelta
import shutil
from pathlib import Path

import sys
sys.path.append("../checks")
from validator_config import SIGMA_MAX, ALPHA_MIN
from image_quality import blur, brightness

SENSOR_COUNT = 2
START_TIME = datetime(2026, 1, 1, 0, 0, 0)
TARGET_SENSOR = "sensor_01"

def assign_dimensions(records):
    # sort by frame_id
    sorted_records = sorted(records, key=lambda r: r["frame_id"])

    assigned_records = []

    for index, record in enumerate(sorted_records):
        sensor_id = f"sensor_{index % SENSOR_COUNT:02d}"
        captured_timestamp = START_TIME + timedelta(seconds=index)

        new_record = {
            "frame_id": record["frame_id"],
            "sensor_id": sensor_id,
            "captured_timestamp": captured_timestamp,
        }
        assigned_records.append(new_record)

    return assigned_records

def assign_severity(dimension_rows, target_sensor):
    '''
    add drift severity to each row IN PLACE. modify the input and return it. Only to the sensor ramps; the rest is 0.0
    '''
    targeted_rows = [r for r in dimension_rows if r["sensor_id"] == target_sensor]
    targeted_rows = sorted(targeted_rows, key=lambda r: r["captured_timestamp"])

    n_row = len(targeted_rows)

    for index, row in enumerate(targeted_rows):
        if n_row == 1:
            row["drift_severity"] = 0.0
        else:
            row["drift_severity"] = index / (n_row - 1)

    other_rows = [r for r in dimension_rows if r["sensor_id"] != target_sensor]

    for row in other_rows:
        row["drift_severity"] = 0.0

    return dimension_rows

def degrade(img, severity):
    '''
    input image and the severity, and return a degraded version
    '''
    sigma = severity * SIGMA_MAX
    alpha = 1 - severity * (1 -ALPHA_MIN)

    if sigma > 0:
        blurred_img = cv2.GaussianBlur(img, (0, 0), sigma)
    else:
        blurred_img = img
    new_img = cv2.convertScaleAbs(blurred_img, alpha=alpha, beta=0)
    return new_img

class DriftSimulator:
    '''
    Degrade a copy of a CLEAN dataset.
    '''
    def __init__(
            self,
            dir_pairs, # [(img_dir, lbl_dir), ...] ; one pair per split
            output_dir,
            sensor_count = SENSOR_COUNT,
            target_sensor = TARGET_SENSOR,
            start_time = START_TIME,
    ):
        self.dir_pairs = [(Path(i), Path(l)) for i, l in dir_pairs]
        self.output_dir = Path(output_dir)
        self.sensor_count = sensor_count
        self.target_sensor = target_sensor
        self.start_time = start_time
    def run(self):
        '''
        Degrade every frames. Return the rows' dimensions.
        '''
        records = []
        lbl_dir_by_frame = {}
        for img_dir, lbl_dir in self.dir_pairs:
            part = loader(img_dir, lbl_dir)
            records.extend(part)
            for r in part:
                lbl_dir_by_frame[r["frame_id"]] = lbl_dir

        dimension_rows = assign_dimensions(records)
        dimension_rows = assign_severity(dimension_rows, self.target_sensor)

        path_by_frame = {r["frame_id"]: r["image_path"] for r in records}

        out_img_dir = self.output_dir / "images"
        out_lbl_dir = self.output_dir / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for row in dimension_rows:
            frame_id = row["frame_id"]

            img = cv2.imread(str(path_by_frame[frame_id]))
            if img is None:
                raise FileNotFoundError(path_by_frame[frame_id])

            degraded = degrade(img, row["drift_severity"])

            out_img_path = out_img_dir / f"{frame_id}.jpg"
            if not cv2.imwrite(str(out_img_path), degraded):
                raise IOError(f"imwrite failed: {out_img_path}")

            shutil.copy(
                lbl_dir_by_frame[frame_id] / f"{frame_id}.txt",
                out_lbl_dir / f"{frame_id}.txt",
            )

        return dimension_rows


if __name__ == "__main__":
    fake = [{"frame_id": f"frame_{i}"} for i in range(8)]

    # assign dimension test
    print("Dimension test")
    for row in assign_dimensions(fake):
        print(row)

    # assign severity test
    print("\nSeverity test")
    rows = assign_dimensions(fake)
    for row in assign_severity(rows, TARGET_SENSOR):
        print(row)

    # assign seversity one row test
    print("\nOne row test")
    fake_two = [{"frame_id": f"frame_{i}"} for i in range(2)]
    rows = assign_dimensions(fake_two)
    for row in assign_severity(rows, TARGET_SENSOR):
        print(row)

    # degrade function test
    test_path = "../datasets/coco8/images/train/000000000009.jpg"
    img = cv2.imread(test_path)
    if img is None:
        raise FileNotFoundError(test_path)

    print("\nDegrade test")
    for severity in [0.0, 0.33, 0.67, 1.0]:
        out = degrade(img, severity)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        print(
            f"severity {severity:.2f}  "
            f"blur {blur(gray)['metric']:8.2f}  "
            f"brightness {brightness(gray)['metric']:6.2f}"
        )