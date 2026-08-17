# this file is to test out the sigma values that change the blur score in a useful way.
import cv2
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

#sweep experiement for blur

# for path in paths:
#     print(path)
#     img = cv2.imread(path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     sigma_value = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
#
#     blur_values = []
#     for sigma in sigma_value:
#         if sigma == 0:
#             result = blur(gray)
#         else:
#             blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
#             result = blur(blurred)
#
#         blur_values.append(result["metric"])
#
#     sigma_dict = {
#         "sigma": sigma_value,
#         "blur_score": blur_values,
#     }
#
#     print(pd.DataFrame(sigma_dict))

# sweep experiment for brightness

for path in paths:
    print(path)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    factor_value = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    brightness_values = []
    for factor in factor_value:
        darker = cv2.convertScaleAbs(gray, alpha=factor, beta=0)
        result = brightness(darker)
        brightness_values.append(result["metric"])

    factor_dict = {
        "factor_value": factor_value,
        "brightness_scores": brightness_values,
    }

    print(pd.DataFrame(factor_dict))
