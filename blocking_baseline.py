import pandas as pd
import re
import unicodedata
from collections import defaultdict

TRAIN = "dataset/train"


def normalize(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


print("Loading data...")

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

print("Normalizing fields...")

for df in [s1, s2, s3]:
    df["name_norm"] = df["business_name"].map(normalize)
    df["address_norm"] = df["business_address"].map(normalize)
    df["country_norm"] = df["country"].map(normalize)

print("Building blocking indexes...")


def build_index(df, column):
    index = defaultdict(list)

    for entity_id, value in zip(df["entity_id"], df[column]):
        if value:
            index[value].append(entity_id)

    return index


name2 = build_index(s2, "name_norm")
name3 = build_index(s3, "name_norm")

address2 = build_index(s2, "address_norm")
address3 = build_index(s3, "address_norm")


s1_lookup = s1.set_index("entity_id")


print("Evaluating blocking recall...")

total_true = 0
found_name = 0
found_address = 0
found_union = 0

for source1_id, matched in zip(
    gt["source1_entity_id"],
    gt["matched_entity_ids"]
):

    true_ids = set(matched.split(",")) if matched else set()

    row = s1_lookup.loc[source1_id]

    name = row["name_norm"]
    address = row["address_norm"]

    name_candidates = set(name2.get(name, []))
    name_candidates.update(name3.get(name, []))

    address_candidates = set(address2.get(address, []))
    address_candidates.update(address3.get(address, []))

    union_candidates = name_candidates | address_candidates

    total_true += len(true_ids)

    found_name += len(true_ids & name_candidates)
    found_address += len(true_ids & address_candidates)
    found_union += len(true_ids & union_candidates)


print()
print("=" * 60)
print("BLOCKING RESULTS")
print("=" * 60)

print("Total true matches:", total_true)

print(
    "Name-only recall:",
    round(found_name / total_true, 6)
)

print(
    "Address-only recall:",
    round(found_address / total_true, 6)
)

print(
    "Name + Address recall:",
    round(found_union / total_true, 6)
)