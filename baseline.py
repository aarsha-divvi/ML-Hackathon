import pandas as pd
import re
import unicodedata

TRAIN = "dataset/train"


def normalize(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


print("Loading files...")

s1 = pd.read_csv(
    f"{TRAIN}/train_source1.tsv",
    sep="\t",
    dtype=str
)

s2 = pd.read_csv(
    f"{TRAIN}/train_source2.tsv",
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    f"{TRAIN}/train_source3.tsv",
    sep="\t",
    dtype=str
)

gt = pd.read_csv(
    f"{TRAIN}/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("Creating normalized names...")

s1["name_norm"] = s1["business_name"].map(normalize)
s2["name_norm"] = s2["business_name"].map(normalize)
s3["name_norm"] = s3["business_name"].map(normalize)

print("Building indexes...")

name2 = {}

for entity_id, name in zip(s2["entity_id"], s2["name_norm"]):
    if name:
        name2.setdefault(name, []).append(entity_id)

name3 = {}

for entity_id, name in zip(s3["entity_id"], s3["name_norm"]):
    if name:
        name3.setdefault(name, []).append(entity_id)

s1_names = dict(
    zip(s1["entity_id"], s1["name_norm"])
)

print("Evaluating exact normalized-name matching...")

total_true = 0
found_true = 0

for source1_id, matched in zip(
    gt["source1_entity_id"],
    gt["matched_entity_ids"]
):

    if matched:
        true_ids = set(matched.split(","))
    else:
        true_ids = set()

    name = s1_names.get(source1_id, "")

    predicted = set(name2.get(name, []))
    predicted.update(name3.get(name, []))

    total_true += len(true_ids)
    found_true += len(true_ids & predicted)


print()
print("=" * 50)
print("RESULT")
print("=" * 50)

print("Total true matches:", total_true)
print("Recovered by exact normalized name:", found_true)

if total_true:
    print(
        "Recall:",
        round(found_true / total_true, 6)
    )