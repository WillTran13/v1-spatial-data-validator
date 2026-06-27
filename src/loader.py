from pathlib import Path

def loader(img_dir, label_dir):
    loader_ls = []
    img_path = Path(img_dir)
    lb_path = Path(label_dir)
    lb_name = lb_path.glob("*.txt")

    for lb_file in list(lb_name):
        lb_text = lb_file.read_text()
        boxes = []
        img_file_path = img_path/(lb_file.stem + ".jpg")

        for line in lb_text.split("\n"):
            if not line:
                continue
            line_ls = line.split(" ")
            line_ls[0] = int(line_ls[0])
            line_ls[1:] = [float(x) for x in line_ls[1:]]
            boxes.append(line_ls)

        img_dict = {}
        img_dict["frame_id"] = lb_file.stem
        img_dict["image_path"] = img_file_path
        img_dict["boxes"] = boxes

        loader_ls.append(img_dict)

    return loader_ls

if __name__ == "__main__":
    train_records = loader("../datasets/coco8/images/train", "../datasets/coco8/labels/train")
    val_records = loader("../datasets/coco8/images/val", "../datasets/coco8/labels/val")

    final_records = train_records + val_records

    print(len(final_records))
    print(final_records[0]["image_path"].exists())