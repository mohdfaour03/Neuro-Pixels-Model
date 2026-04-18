import glob

for f in glob.glob("src/*.py"):
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
    # Replace escaped quotes with proper quotes
    content = content.replace('\\"\\"\\"', '"""')
    with open(f, "w", encoding="utf-8") as file:
        file.write(content)

print("Fixed syntax errors.")
