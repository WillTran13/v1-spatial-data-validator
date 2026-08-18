# return dict: {"metric": n number of bad boxes, "status": True if n bad box is 0}
def out_of_bounds(boxes):
    counter = 0
    for box in boxes:
        cls, xc, yc, w, h = box
        if xc - w/2 < 0 or yc - h/2 < 0 or xc + w/2 > 1 or yc + h/2 > 1:
            counter += 1

    status = counter == 0

    return {"metric": counter, "status": status}

def zero_area(boxes):
    counter = 0
    for box in boxes:
        cls, xc, yc, w, h = box
        if w <= 0 or h <= 0:
            counter += 1

    status = counter == 0

    return {"metric": counter, "status": status}

def invalid_class(boxes, num_classes=80):
    counter = 0
    for box in boxes:
        cls, xc, yc, w, h = box
        if cls < 0 or cls >= num_classes:
            counter += 1
    status = counter == 0

    return {"metric": counter, "status": status}


if __name__ == "__main__":
    import sys

    sys.path.append("../src")

    from loader import loader
    # test 1, clean
    records = loader("../datasets/coco8/images/train", "../datasets/coco8/labels/train")
    result = out_of_bounds(records[0]["boxes"])
    print(result)
    assert result["metric"] == 0, "this has bad boxes"

    # test 2, bad boxes but look valid
    bad_box = [[5, 0.95, 0.5, 0.2, 0.1]]
    result = out_of_bounds(bad_box)
    assert result["metric"] == 1, "the result should be bad, box spilling"

    # zero area
    assert zero_area([[5, 0.5, 0.5, 0.0, 0.2]])["metric"] == 1, "0 width"
    assert zero_area(records[0]["boxes"])["metric"] == 0, "clean frame flagged"

    # invalid class
    assert invalid_class([[80, 1, 1, 1, 1]])["metric"] == 1, "class 80 not caught"
    assert invalid_class([[79, 1, 1, 1, 1]])["metric"] == 0, "class 79 is valid"

    print("passed")

