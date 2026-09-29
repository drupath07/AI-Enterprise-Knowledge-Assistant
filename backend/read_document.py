from pathlib import Path

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Find the documents folder
file_path = project_folder / "documents" / "company_policy.txt"

# Read the document
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("Document content:")
print(text)