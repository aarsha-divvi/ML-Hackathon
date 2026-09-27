import pandas as pd

TRAIN = "dataset/train"

print("Loading ground truth...")

gt = pd.read_csv(
    f"{TRAIN}/train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

print("Ground truth shape:", gt.shape)

print("\nNumber of matches per Source 1 entity:")

counts = gt["matched_entity_ids"].fillna("").apply(
    lambda x: 0 if not x else len(x.split(","))
)

print(counts.describe())

print("\nEntities with no matches:", (counts == 0).sum())
print("Entities with 1 match:", (counts == 1).sum())
print("Entities with >1 matches:", (counts > 1).sum())

print("\nExample ground truth:")
print(gt.head(10).to_string(index=False))