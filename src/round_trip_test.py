'''
1 frame round trip test:
degrade, save, reload, run_checks

this is a test run to see if the pipeline can read a file before building the real simulator.
'''

from loader import loader
from run_checks import run_checks

import shutil
from pathlib import Path

import cv2

FRAME_ID = "000000000009"
SIGMA = 1.3  # sigma_max from sweep_experiment

CLEAN_IMG_DIR = Path("../datasets/coco8/images/train")
CLEAN_LBL_DIR = Path("../datasets/coco8/labels/train")
DEG_IMG_DIR = Path("../data/degraded/images")
DEG_LBL_DIR = Path("../data/degraded/labels")

def main():
    # make the folder if not exist
    DEG_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DEG_LBL_DIR.mkdir(parents=True, exist_ok=True)

    # load image
    src_img_path = CLEAN_IMG_DIR / f"{FRAME_ID}.jpg"
    img = cv2.imread(str(src_img_path))
    if img is None:
        raise FileNotFoundError(src_img_path)

    # blur image
    degraded = cv2.GaussianBlur(img, (0, 0), SIGMA)

    # save the save file name for comparison later
    out_img_path = DEG_IMG_DIR / f"{FRAME_ID}.jpg"
    if not cv2.imwrite(str(out_img_path), degraded):
        raise IOError(f"imwrite failed: {out_img_path}")

    # copy the label file, loader function will use it
    shutil.copy(CLEAN_LBL_DIR / f"{FRAME_ID}.txt", DEG_LBL_DIR / f"{FRAME_ID}.txt")

    # user loader and run_checks
    degraded_records = loader(DEG_IMG_DIR, DEG_LBL_DIR)
    degraded_df = run_checks(degraded_records, run_type="degraded")

    # compare with clean row
    clean_records = loader(CLEAN_IMG_DIR, CLEAN_LBL_DIR)
    clean_df = run_checks(clean_records, run_type="clean")
    clean_row = clean_df[clean_df["frame_id"] == FRAME_ID]

    # report
    print("clean row:")
    print(clean_row)
    print("\ndegraded row:")
    print(degraded_df)

    print("records loaded:", len(degraded_records))
    print("frame_id matches:", degraded_df["frame_id"].iloc[0] == FRAME_ID)
    print("clean blur:", clean_row["blur_score"].iloc[0])
    print("degraded blur:", degraded_df["blur_score"].iloc[0])

if __name__ == "__main__":
    main()
