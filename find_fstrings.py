import os
import tokenize

for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".py") and not file.startswith("."):
            path = os.path.join(root, file)
            with open(path, "rb") as f:
                try:
                    for tok in tokenize.tokenize(f.readline):
                        if tok.type == tokenize.STRING:
                            s = tok.string
                            if s.startswith(
                                ('f"', "f'", 'F"', "F'")
                            ) and not s.startswith(('f"""', "f'''", 'F"""', "F'''")):
                                if "\n" in s:
                                    print(f"File {path} line {tok.start[0]}: {s}")
                except Exception:
                    pass
