import re
import unicodedata


# ---------------------------------------------------------
# General text normalization
# ---------------------------------------------------------

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).strip().lower()

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Replace common separators with spaces
    value = re.sub(r"[/,_\-]+", " ", value)

    # Remove remaining punctuation
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ---------------------------------------------------------
# Business name normalization
# ---------------------------------------------------------

def normalize_business_name(value):

    value = normalize_text(value)

    # Common legal/business suffix normalization
    replacements = {
        "private limited": "pvt ltd",
        "private ltd": "pvt ltd",
        "limited": "ltd",
        "incorporated": "inc",
        "corporation": "corp",
        "company": "co",
        "llc": "llc",
        "l l c": "llc",
    }

    words = value.split()

    normalized_words = []

    for word in words:
        normalized_words.append(replacements.get(word, word))

    return " ".join(normalized_words)


# ---------------------------------------------------------
# Address normalization
# ---------------------------------------------------------

def normalize_address(value):

    value = normalize_text(value)

    replacements = {
        "street": "st",
        "road": "rd",
        "avenue": "ave",
        "boulevard": "blvd",
        "drive": "dr",
        "lane": "ln",
        "parkway": "pkwy",
        "highway": "hwy",
        "apartment": "apt",
        "suite": "ste",
    }

    words = value.split()

    normalized_words = []

    for word in words:
        normalized_words.append(replacements.get(word, word))

    return " ".join(normalized_words)