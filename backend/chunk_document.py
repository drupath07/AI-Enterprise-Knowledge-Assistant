from pathlib import Path

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Find the document
file_path = project_folder / "documents" / "company_policy.txt"

# Read the document
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Split the document into paragraphs
chunks = text.split("\n\n")

print("Number of chunks:", len(chunks))

print("\n--- CHUNKS ---")

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)