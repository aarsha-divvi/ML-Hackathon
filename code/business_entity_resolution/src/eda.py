import pandas as pd
import os

# ---------------------------------------------------------
# Location of the training data
# ---------------------------------------------------------

BASE = BASE = "student_resource/dataset/train"

files = {
    "SOURCE 1": "train_source1.tsv",
    "SOURCE 2": "train_source2.tsv",
    "SOURCE 3": "train_source3.tsv",
    "GROUND TRUTH": "train_ground_truth.tsv"
}


# ---------------------------------------------------------
# Read and inspect each file
# ---------------------------------------------------------

for name, filename in files.items():

    path = os.path.join(BASE, filename)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # Read TSV
    df = pd.read_csv(path, sep="\t")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))