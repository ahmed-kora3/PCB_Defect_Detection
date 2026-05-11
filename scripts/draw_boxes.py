import argparse
import os
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Draw bounding boxes from YOLO-style label files on PCB images.")
parser.add_argument("--split", choices=["train", "val", "test"], default="train",
                    help="Dataset split to use")
parser.add_argument("--label-file", default=None,
                    help="Label filename to draw (default: first label file in directory)")
parser.add_argument("--image-ext", default=".jpg",
                    help="Image file extension to search for")
args = parser.parse_args()

splits = ["train", "val", "test"]
image_path = os.path.join("dataset", "pcb-defect-dataset", args.split, "images")
label_path = os.path.join("dataset", "pcb-defect-dataset", args.split, "labels")

if not os.path.isdir(label_path):
    raise FileNotFoundError(f"Label directory not found: {label_path}")
if not os.path.isdir(image_path):
    raise FileNotFoundError(f"Image directory not found: {image_path}")

label_files = sorted([f for f in os.listdir(label_path) if f.lower().endswith(".txt")])
if not label_files:
    raise FileNotFoundError(f"No label files found in: {label_path}")

label_file = args.label_file if args.label_file else label_files[0]
label_file_path = None
if label_file not in label_files:
    found = False
    for other_split in splits:
        if other_split == args.split:
            continue
        candidate_label_path = os.path.join("dataset", "pcb-defect-dataset", other_split, "labels")
        if os.path.isdir(candidate_label_path) and label_file in os.listdir(candidate_label_path):
            label_path = candidate_label_path
            image_path = os.path.join("dataset", "pcb-defect-dataset", other_split, "images")
            if not os.path.isdir(image_path):
                raise FileNotFoundError(f"Image directory not found for split {other_split}: {image_path}")
            label_file_path = os.path.join(label_path, label_file)
            found = True
            print(f"Using label file from split '{other_split}'")
            break
    if not found:
        raise FileNotFoundError(f"Label file not found: {label_file}")
else:
    label_file_path = os.path.join(label_path, label_file)

assert label_file_path is not None
label_stem = os.path.splitext(label_file)[0]

candidates = {label_stem}
if label_stem.endswith("_256"):
    candidates.add(label_stem[:-4] + "_600")
    candidates.add(label_stem[:-4])
    candidates.add(label_stem.replace("_256", ""))
    candidates.add(label_stem.replace("_256", "_600"))

image_file = None
for img in os.listdir(image_path):
    name, ext = os.path.splitext(img)
    if ext.lower() != args.image_ext.lower():
        continue
    if name in candidates:
        image_file = img
        break

if image_file is None:
    print("No matching image found for:", label_file)
    print("Tried candidates:", sorted(candidates))
    raise FileNotFoundError("Matching image file not found")

img_file_path = os.path.join(image_path, image_file)
image = cv2.imread(img_file_path)
if image is None:
    raise RuntimeError(f"Failed to load image: {img_file_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w, _ = image.shape
with open(label_file_path, "r") as f:
    labels = [line.strip() for line in f if line.strip()]

for label in labels:
    class_id, x_center, y_center, width, height = map(float, label.split())
    x_center *= w
    y_center *= h
    width *= w
    height *= h
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(8,8))
plt.imshow(image)
plt.axis("off")
plt.title(f"PCB Defect Detection - {label_file}")
plt.show()