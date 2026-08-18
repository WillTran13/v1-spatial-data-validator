# Every check returns a dict: {"metric": metric, "status": True for passed/False for failed}. This dict is applied for all function outputs
import cv2
from config import THRESHOLDS


def blur(gray, threshold=THRESHOLDS["blur_min"]):
    '''
    Check the image and output metric of blurriness
    Blur metric: variance of the Laplacian. Expects a single channel grayscale array
    '''

    # compute laplacian in signed float
    lap_float = cv2.Laplacian(gray, cv2.CV_64F)

    # take the variance
    lap_var = lap_float.var()

    # compare with threshold
    blur_status = lap_var >= threshold

    # return metric
    return {"metric": lap_var, "status": blur_status}

def brightness(gray, min_threshold=THRESHOLDS["brightness_min"], max_threshold=THRESHOLDS["brightness_max"]):

    # take the mean
    bright_mean = gray.mean()

    # compare with threshold
    bright_status = bright_mean >= min_threshold and bright_mean <= max_threshold

    # return metric
    return {"metric": bright_mean, "status": bright_status}

def resolution(img, min_side=THRESHOLDS["resolution_min"]):

    # get the height and width
    height, width = img.shape[:2]

    # get the min of the metrics
    min_resolution = min(height, width)

    # compare with min_side
    resolution_status = min_resolution >= min_side

    # return the value
    return {"metric": min_resolution, "status": resolution_status}

def contrast(gray, min_threshold=THRESHOLDS["contrast_min"]):

    # take the mean
    contrast_std = gray.std()

    # compare with threshold
    contrast_status = contrast_std >= min_threshold

    # return metric
    return {"metric": contrast_std, "status": contrast_status}


if __name__ == "__main__":
    img = cv2.imread("../datasets/coco8/images/train/000000000009.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # print all metrics of the image
    print("blur ", blur(gray)["metric"])
    print("brightness ", brightness(gray)["metric"])
    print("contrast ", contrast(gray)["metric"])
    print("resolution ", resolution(img)["metric"])

    # blur: blurrer -> lower score
    blurred = blur(cv2.GaussianBlur(gray, (9,9), 0))
    assert blurred["metric"] < blur(gray)["metric"], "blur did not drop"

    # brightness: darker -> score lower
    dark = cv2.convertScaleAbs(gray, alpha=0.4, beta=0)
    assert brightness(dark)["metric"] < brightness(gray)["metric"], "brightness did not drop"

    # resolution: direct check
    assert resolution(img)["metric"] == 480, "resolution incorrect"

    # constrat: flatten spread, keep mean
    mean_value = gray.mean()
    low_contrast = (mean_value + (gray - mean_value) * 0.3).astype("uint8")
    assert contrast(low_contrast)["metric"] < contrast(gray)["metric"], "contrast did not drop"
    assert abs(brightness(low_contrast)["metric"] - brightness(gray)["metric"]) < 5, "brightness should stay flat when only contrast change"

    print("All checks passed")

