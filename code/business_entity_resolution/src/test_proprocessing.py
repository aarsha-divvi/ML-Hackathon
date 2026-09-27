from preprocessing import (
    normalize_text,
    normalize_business_name,
    normalize_address
)


examples = [
    "Amazon India Pvt. Ltd.",
    "AMAZON INDIA PRIVATE LIMITED",
    "Christ Chapel",
    "Orelee's Barbershop",
    "B+ Retail Inc"
]

print("\nBUSINESS NAME TESTS")
print("=" * 60)

for x in examples:
    print(f"{x:40} -> {normalize_business_name(x)}")


addresses = [
    "2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD",
    "17560 Ellis Road, Tahlequah, OK",
    "1795 Westchester Drive, High Point, NC"
]

print("\nADDRESS TESTS")
print("=" * 60)

for x in addresses:
    print(f"{x:60} -> {normalize_address(x)}")