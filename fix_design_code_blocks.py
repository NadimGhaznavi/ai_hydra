#!/usr/bin/env python3
"""
Fix Python code blocks in design.rst to have proper syntax.
"""

import re

def fix_design_rst():
    """Fix code blocks in design.rst to have proper Python syntax."""
    
    with open('docs/_source/design.rst', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match code blocks
    code_block_pattern = r'(\.\. code-block:: python\n\n)(.*?)(?=\n\S|\n\.\.|$)'
    
    def fix_code_block(match):
        prefix = match.group(1)
        code = match.group(2)
        
        # Split into lines
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix method definitions without colons
            if re.match(r'(\s+)(def|async def)\s+\w+.*\)\s*->\s*\w+\s*$', line):
                line = line.rstrip() + ':'
                fixed_lines.append(line)
                # Add pass statement with proper indentation
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(' ' * (indent + 4) + '"""Method implementation."""')
                fixed_lines.append(' ' * (indent + 4) + 'pass')
                fixed_lines.append('')
            elif re.match(r'(\s+)(def|async def)\s+\w+.*\)\s*$', line):
                line = line.rstrip() + ':'
                fixed_lines.append(line)
                # Add pass statement with proper indentation
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(' ' * (indent + 4) + '"""Method implementation."""')
                fixed_lines.append(' ' * (indent + 4) + 'pass')
                fixed_lines.append('')
            elif re.match(r'(\s+)def __init__.*\)\s*$', line):
                line = line.rstrip() + ':'
                fixed_lines.append(line)
                # Add pass statement with proper indentation
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(' ' * (indent + 4) + '"""Initialize instance."""')
                fixed_lines.append(' ' * (indent + 4) + 'pass')
                fixed_lines.append('')
            else:
                fixed_lines.append(line)
        
        return prefix + '\n'.join(fixed_lines)
    
    # Apply fixes
    fixed_content = re.sub(code_block_pattern, fix_code_block, content, flags=re.DOTALL)
    
    # Write back
    with open('docs/_source/design.rst', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("✅ Fixed Python code blocks in design.rst")

if __name__ == "__main__":
    fix_design_rst()