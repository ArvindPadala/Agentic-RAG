import os
import re

files_to_check = [
    "ade_s3_handler.py",
    "agent.py",
    "lambda_helpers.py",
    "upload_handler.py",
    "visual_grounding_helper.py",
    "query_optimizer.py"
]

for filename in files_to_check:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
        
        # Regex to find f"..." or f'...' with a newline inside
        # We find: f" (anything not quote) \n (whitespace) (anything not quote) "
        def repl(m):
            s = m.group(0)
            return s.replace('\n', ' ').replace('        ', ' ')
            
        content = re.sub(r'f"([^"\n]*\n[^"]*)"', repl, content)
        content = re.sub(r"f'([^'\n]*\n[^']*)'", repl, content)
        
        # also for multiple newlines
        content = re.sub(r'f"([^"\n]*\n[^"\n]*\n[^"]*)"', repl, content)
        content = re.sub(r"f'([^'\n]*\n[^'\n]*\n[^']*)'", repl, content)
        
        with open(filename, 'w') as f:
            f.write(content)
