
from pathlib import Path
import pandas as pd
import zipfile

project = Path(__file__).resolve().parent
dataset = project / "data" / "SkinDisNet"
images_root = dataset / "Preprocessed"

zip_path = project / "eczema_training_data.zip"

splits = ["train", "val", "test"]

with zipfile.ZipFile(
    zip_path, "w", zipfile.ZIP_DEFLATED
) as zip_file:

    for split in splits:
        csv_path = dataset / f"{split}.csv"
        df = pd.read_csv(csv_path)

        portable_paths = []

        for _, row in df.iterrows():
            # Convert Windows image path to a Path object
            original_path = Path(row["image_path"])

            # Extract folder and filename
            folder_name = original_path.parent.name
            filename = original_path.name

            # Build path to the image
            image_path = images_root / folder_name / filename

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing image: {image_path}"
                )

            # Path used inside ZIP and Colab
            relative_path = (
                f"images/{folder_name}/{filename}"
            )

            portable_paths.append(relative_path)

            # Add the image to the ZIP
            zip_file.write(
                image_path,
                arcname=relative_path
            )

        # Replace Windows paths with portable paths
        df["image_path"] = portable_paths

        # Save a temporary portable CSV
        temp_csv = dataset / f"{split}_portable.csv"
        df.to_csv(temp_csv, index=False)

        # Add CSV to ZIP
        zip_file.write(
            temp_csv,
            arcname=f"{split}.csv"
        )

        temp_csv.unlink()

        print(f"{split}: {len(df)} records added")

print("\nZIP created successfully!")
print(zip_path)
