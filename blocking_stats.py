import pandas as pd
import re
import unicodedata
from collections import Counter

TRAIN = "dataset/train"


def normalize(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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

print("Normalizing...")

for df in [s1, s2, s3]:
    df["name_norm"] = df["business_name"].map(normalize)
    df["address_norm"] = df["business_address"].map(normalize)


def first_token(value):
    if not value:
        return ""
    return value.split()[0]


def first_two_tokens(value):
    if not value:
        return ""
    parts = value.split()
    return " ".join(parts[:2])


def prefix(value, n):
    if not value:
        return ""
    return value[:n]


for df in [s1, s2, s3]:
    df["name_first"] = df["name_norm"].map(first_token)
    df["name_first2"] = df["name_norm"].map(first_two_tokens)
    df["name_prefix4"] = df["name_norm"].map(lambda x: prefix(x, 4))
    df["name_prefix6"] = df["name_norm"].map(lambda x: prefix(x, 6))


print("\nCalculating key frequencies...")


def show_frequency(df, column):
    counts = df[column].value_counts()

    counts = counts[counts.index != ""]

    print("\n", column)
    print("Unique keys:", len(counts))
    print("Median records/key:", round(counts.median(), 2))
    print("Mean records/key:", round(counts.mean(), 2))
    print("95th percentile:", round(counts.quantile(.95), 2))
    print("99th percentile:", round(counts.quantile(.99), 2))
    print("Maximum records/key:", counts.max())

    print("Top 10:")
    print(counts.head(10).to_string())


for column in [
    "name_first",
    "name_first2",
    "name_prefix4",
    "name_prefix6"
]:
    show_frequency(s2, column)
    show_frequency(s3, column)