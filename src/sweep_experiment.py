# this file is to test out the sigma values that change the blur score in a useful way.
import cv2
from pathlib import Path
import sys
import pandas as pd

sys.path.append("../checks")
from image_quality import blur, brightness

paths = [
    "../datasets/coco8/images/train/000000000009.jpg",
    "../datasets/coco8/images/train/000000000025.jpg",
    "../datasets/coco8/images/train/000000000030.jpg",
    "../datasets/coco8/images/train/000000000034.jpg"
]

def load_gray(path):
    ''' Read image as grayscale'''
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("Image not found")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sweep_blur(paths, sigma_values):
    '''Blur each image at each sigma'''
    rows = []

    for path in paths:
        gray = load_gray(path)
        clean = blur(gray)["metric"]
        for sigma in sigma_values:
            result = blur(cv2.GaussianBlur(gray, (0,0), sigma))
            rows.append({
                "frame_id": Path(path).stem,
                "knob": sigma, # change to knob to work with first_failure filter
                "metric": result["metric"],
                "clean": clean,
                "ratio": result["metric"] / clean,
                "status": result["status"],
            })

    return pd.DataFrame(rows)



def sweep_brightness(paths, factor_values):
    '''Darken each image at each factor'''
    rows = []
    for path in paths:
        gray = load_gray(path)
        clean = brightness(gray)["metric"]
        for factor in factor_values:
            result = brightness(cv2.convertScaleAbs(gray, alpha=factor, beta=0))
            rows.append({
                "frame_id": Path(path).stem,
                "knob": factor, # change to knob to work with first_failure filter
                "metric": result["metric"],
                "clean": clean,
                "ratio": result["metric"] / clean,
                "status": result["status"],
            })

    return pd.DataFrame(rows)


def first_failure(df):
    '''The first sigma/factor setting per frame where the check flips to FAIL'''
    failed = df[~df["status"]]
    return failed.groupby("frame_id").first()[["knob", "metric"]]

if __name__ == "__main__":
    sigmas = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    blur_df = sweep_blur(paths, sigmas)
    bright_df = sweep_brightness(paths, factors)

    print("blur crossings")
    print(first_failure(blur_df))
    print("\nbrightness crossings")
    print(first_failure(bright_df))
