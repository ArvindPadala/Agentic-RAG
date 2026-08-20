import os
import re

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.py') and not file.startswith('.'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            
            # This regex looks for:
            # f"...{ followed by optional whitespace and newline, then more whitespace, then something ending in }# Actually, let's just find lines ending with { and join them with the next linelines = content.split('\n')new_lines = []i = 0while i < len(lines):line = lines[i]if ('f"' in line or "f'" in line) and line.rstrip().endswith('{'):
                    # join with next line
                    next_line = lines[i+1].lstrip() if i+1 < len(lines) else ""
                    line = line.rstrip() + next_line
                    i += 1
                new_lines.append(line)
                i += 1
            
            with open(path, 'w') as f:
                f.write('\n'.join(new_lines))
