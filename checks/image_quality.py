# Every check returns a dict: {"metric": metric, "status": True for passed/False for failed}. This dict is applied for all function outputs
import cv2

def blur(gray, threshold=100): # provisional - get the real threshold number in sess3
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


if __name__ == "__main__":
    img = cv2.imread("../datasets/coco8/images/train/000000000009.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharp = blur(gray)
    blurred = blur(cv2.GaussianBlur(gray, (9, 9), 0))

    print(sharp["metric"])
    print(blurred["metric"])

    assert blurred["metric"] < sharp["metric"], "Wrong direction, blur metric did not drop"
    print("PASSED, blurrier image score lower")
