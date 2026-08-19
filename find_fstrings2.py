import glob

for filename in glob.glob("**/*.py", recursive=True):
    with open(filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # check if line has 'f"' or "f'" and ends with {if 'f"' in line or "f'" in line:
                if line.rstrip().endswith("{"):
                    print(f"{filename}:{i+1}: {line.strip()}")
