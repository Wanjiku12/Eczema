
import re
from pathlib import Path
import pandas as pd

# Project folders
project = Path(__file__).resolve().parent
dataset = project / "data" / "SkinDisNet"

metadata_path = dataset / "SkinDisNet_Metadata.csv"
images_root = dataset / "Preprocessed"

df = pd.read_csv(metadata_path)

# Supported image extensions
extensions = [".jpg", ".jpeg", ".png", ".webp"]

def find_image(row):
    # Extract folder code: e.g. "Atopic Dermatitis (AD)" -> "AD"
    folder_match = re.search(r"\((\w+)\)", str(row["Folder_name"]))

    # Extract image number: e.g. "AD (1)" -> 1
    number_match = re.search(r"\((\d+)\)", str(row["Image_id"]))

    if not folder_match or not number_match:
        return None

    code = folder_match.group(1)
    number = int(number_match.group(1))

    folder = images_root / code

    # Match image names such as AD_01.jpg or AD_1.jpg
    for ext in extensions:
        for filename in [
            f"{code}_{number:02d}{ext}",
            f"{code}_{number}{ext}",
            f"{code}_{number:03d}{ext}"
        ]:
            path = folder / filename

            if path.is_file():
                return str(path.resolve())

    return None


# Create image paths
df["image_path"] = df.apply(find_image, axis=1)

# Report results
matched = df["image_path"].notna().sum()
missing = df["image_path"].isna().sum()

print("Total metadata records:", len(df))
print("Matched images:", matched)
print("Missing images:", missing)

if missing:
    print("\nExamples of unmatched records:")
    print(
        df[df["image_path"].isna()][
            ["Folder_name", "Image_id"]
        ].head(15).to_string(index=False)
    )
else:
    # Save only when all images have been matched
    output_path = dataset / "image_manifest.csv"
    df.to_csv(output_path, index=False)

    print("\nManifest saved successfully:")
    print(output_path)

    print("\nExample records:")
    print(
        df[["Patient_id", "Diagnosis", "image_path"]]
        .head()
        .to_string(index=False)
    )
python 