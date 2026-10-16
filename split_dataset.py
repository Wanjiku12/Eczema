
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 1. Find the project files
project = Path(__file__).resolve().parent
dataset = project / "data" / "SkinDisNet"

manifest_path = dataset / "image_manifest.csv"
df = pd.read_csv(manifest_path)

# 2. First split: 70% training, 30% remaining
splitter1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, remaining_idx = next(
    splitter1.split(
        df,
        groups=df["Patient_id"]
    )
)

train_df = df.iloc[train_idx].copy()
remaining_df = df.iloc[remaining_idx].copy()

# 3. Second split: remaining data into
# validation and testing groups
splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    splitter2.split(
        remaining_df,
        groups=remaining_df["Patient_id"]
    )
)

val_df = remaining_df.iloc[val_idx].copy()
test_df = remaining_df.iloc[test_idx].copy()

# 4. Check for patient overlap
train_patients = set(train_df["Patient_id"])
val_patients = set(val_df["Patient_id"])
test_patients = set(test_df["Patient_id"])

assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(test_patients)

print("Patient overlap check passed!")

# 5. Display split statistics
for name, subset in [
    ("TRAINING", train_df),
    ("VALIDATION", val_df),
    ("TESTING", test_df)
]:
    print(f"\n{name}")
    print("Images:", len(subset))
    print("Patients:", subset["Patient_id"].nunique())
    print(subset["Diagnosis"].value_counts())

# 6. Save the splits
train_df.to_csv(dataset / "train.csv", index=False)
val_df.to_csv(dataset / "val.csv", index=False)
test_df.to_csv(dataset / "test.csv", index=False)

print("\nDataset splitting complete!")
