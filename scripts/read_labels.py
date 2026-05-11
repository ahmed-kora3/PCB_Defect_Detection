import os

# Labels path
labels_path = "dataset/pcb-defect-dataset/train/labels"

# Get label files
label_files = os.listdir(labels_path)

print("Number of Label Files:", len(label_files))

# Read first label file
first_label = os.path.join(labels_path, label_files[0])

with open(first_label, "r") as file:
    content = file.read()

print("\nLabel Content:\n")
print(content)