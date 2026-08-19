import os
import re

def fix_fstring(match):
    s = match.group(0)
    # remove newlines and extra spaces
    s = re.sub(r'\n\s*', '', s)
    return s

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.py') and not file.startswith('.'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            
            # Match f"..." and f'...' across multiple lines (non-greedy)
            # but only if it doesn't match a triple quote
            # This regex looks for f" or f' followed by anything except the matching quote, including newlines# We must be careful not to match too much.# Simple approach: find f" followed by any characters including \n until the next "# And same for f'
            content = re.sub(r'f"([^"]*\n[^"]*)"', fix_fstring, content)
            content = re.sub(r"f'([^']*\n[^']*)'", fix_fstring, content)
            
            with open(path, 'w') as f:
                f.write(content)
