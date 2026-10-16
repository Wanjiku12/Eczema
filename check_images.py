
from pathlib import Path
import pandas as pd

# Locate project and dataset
project = Path(__file__).resolve().parent
dataset = project / "data" / "SkinDisNet"

csv_path = dataset / "SkinDisNet_Metadata.csv"
images_root = dataset / "Preprocessed"

df = pd.read_csv(csv_path)

print("Checking image folders...\n")

# Display actual image folders
if not images_root.exists():
    print("Preprocessed folder not found!")
else:
    for folder in images_root.iterdir():
        if folder.is_dir():
            images = [
                f for f in folder.rglob("*")
                if f.suffix.lower() in
                [".jpg", ".jpeg", ".png", ".webp"]
            ]
            print(f"{folder.name}: {len(images)} images")

    print("\nExample metadata records:")
    print(
        df[["Folder_name", "Image_id", "Diagnosis"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nExample actual image filenames:")

    image_files = [
        f for f in images_root.rglob("*")
        if f.suffix.lower() in
        [".jpg", ".jpeg", ".png", ".webp"]
    ]

    for image in image_files[:10]:
        print(image.relative_to(images_root))

    print("\nTotal image files:", len(image_files))
