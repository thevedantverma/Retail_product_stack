import pandas as pd
import os
import shutil

CSV_PATH = r"C:\Users\theve\.cache\kagglehub\datasets\mohamedessam3112002\sku-110k\versions\1\SKU110K_fixed\annotations\annotations_train.csv"
IMAGE_FOLDER = r"C:\Users\theve\.cache\kagglehub\datasets\mohamedessam3112002\sku-110k\versions\1\SKU110K_fixed\images"

OUTPUT_IMAGES = "dataset/images/train"
OUTPUT_LABELS = "dataset/labels/train"

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

df = pd.read_csv(CSV_PATH, header=None)

for _, row in df.iterrows():

    image_name = row[0]

    x1 = row[1]
    y1 = row[2]
    x2 = row[3]
    y2 = row[4]

    img_w = row[6]
    img_h = row[7]

    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h

    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    label_path = os.path.join(
        OUTPUT_LABELS,
        image_name.replace(".jpg", ".txt")
    )

    with open(label_path, "a") as f:

        f.write(
            f"0 {x_center} {y_center} {width} {height}\n"
        )

    src_img = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    dst_img = os.path.join(
        OUTPUT_IMAGES,
        image_name
    )

    if not os.path.exists(dst_img):

        shutil.copy(src_img, dst_img)

print("Conversion Done")