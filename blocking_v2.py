import pandas as pd
import re
import unicodedata
from collections import defaultdict

TRAIN = "dataset/train"


def normalize(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text):
    return set(text.split()) if text else set()


def first_meaningful_token(text):
    stop = {
        "the", "private", "limited", "ltd", "llc", "inc",
        "company", "co", "corp", "corporation", "and",
        "dr", "mr", "mrs", "ms", "sri", "shri", "smt"
    }

    for token in text.split():
        if token not in stop and len(token) >= 3:
            return token

    return ""


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

print("Normalizing...")

for df in [s1, s2, s3]:
    df["name_norm"] = df["business_name"].map(normalize)
    df["address_norm"] = df["business_address"].map(normalize)
    df["country_norm"] = df["country"].map(normalize)

    df["name_first_meaningful"] = (
        df["name_norm"].map(first_meaningful_token)
    )

    df["name_first2"] = (
        df["name_norm"]
        .map(lambda x: " ".join(x.split()[:2]))
    )

    df["name_prefix6"] = (
        df["name_norm"].map(lambda x: x[:6])
    )


def build_index(df, column):
    index = defaultdict(list)

    for entity_id, value in zip(df["entity_id"], df[column]):
        if value:
            index[value].append(entity_id)

    return index


print("Building indexes...")

indexes = {}

for source_name, df in [("s2", s2), ("s3", s3)]:

    indexes[source_name] = {
        "name": build_index(df, "name_norm"),
        "name_first_meaningful": build_index(
            df, "name_first_meaningful"
        ),
        "name_first2": build_index(df, "name_first2"),
        "name_prefix6": build_index(df, "name_prefix6"),
        "address": build_index(df, "address_norm"),
    }


s1_lookup = s1.set_index("entity_id")


print("Evaluating blocking recall...")

total_true = 0

found_exact_name = 0
found_exact_address = 0
found_first_meaningful = 0
found_first2 = 0
found_prefix6 = 0
found_union = 0


for source1_id, matched in zip(
    gt["source1_entity_id"],
    gt["matched_entity_ids"]
):

    true_ids = set(matched.split(",")) if matched else set()

    row = s1_lookup.loc[source1_id]

    name = row["name_norm"]
    address = row["address_norm"]
    first_meaningful = row["name_first_meaningful"]
    first2 = row["name_first2"]
    prefix6 = row["name_prefix6"]

    exact_name = set()
    exact_address = set()
    meaningful = set()
    first2_candidates = set()
    prefix_candidates = set()

    for source in ["s2", "s3"]:

        idx = indexes[source]

        exact_name.update(
            idx["name"].get(name, [])
        )

        exact_address.update(
            idx["address"].get(address, [])
        )

        meaningful.update(
            idx["name_first_meaningful"].get(
                first_meaningful, []
            )
        )

        first2_candidates.update(
            idx["name_first2"].get(first2, [])
        )

        prefix_candidates.update(
            idx["name_prefix6"].get(prefix6, [])
        )

    union = (
        exact_name
        | exact_address
        | meaningful
        | first2_candidates
        | prefix_candidates
    )

    total_true += len(true_ids)

    found_exact_name += len(true_ids & exact_name)
    found_exact_address += len(true_ids & exact_address)
    found_first_meaningful += len(
        true_ids & meaningful
    )
    found_first2 += len(
        true_ids & first2_candidates
    )
    found_prefix6 += len(
        true_ids & prefix_candidates
    )
    found_union += len(true_ids & union)


print()
print("=" * 60)
print("BLOCKING V2 RESULTS")
print("=" * 60)

print("Total true matches:", total_true)

print(
    "Exact name recall:",
    round(found_exact_name / total_true, 6)
)

print(
    "Exact address recall:",
    round(found_exact_address / total_true, 6)
)

print(
    "First meaningful token recall:",
    round(found_first_meaningful / total_true, 6)
)

print(
    "First 2 tokens recall:",
    round(found_first2 / total_true, 6)
)

print(
    "6-character prefix recall:",
    round(found_prefix6 / total_true, 6)
)

print(
    "UNION recall:",
    round(found_union / total_true, 6)
)