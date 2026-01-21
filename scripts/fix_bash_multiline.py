#!/usr/bin/env python3
"""
Fix multi-line bash commands in markdown files to use proper line continuation.

This script finds bash code blocks with commands that span multiple lines without
proper backslash continuation and fixes them.
"""

import re
from pathlib import Path
import sys


def fix_bash_block(bash_content: str) -> str:
    """Fix a single bash code block to use proper line continuation."""
    lines = bash_content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts a command that continues on the next line
        # Pattern: command followed by indented options on next lines
        if line.strip() and not line.strip().startswith('#'):
            # Look ahead to see if next lines are indented continuation
            continuation_lines = []
            j = i + 1
            
            while j < len(lines):
                next_line = lines[j]
                # Check if next line is indented (continuation) and starts with --
                if next_line.startswith('    ') and next_line.strip().startswith('--'):
                    continuation_lines.append(next_line.strip())
                    j += 1
                elif next_line.strip() == '':
                    # Empty line, stop looking
                    break
                else:
                    # Not a continuation
                    break
            
            if continuation_lines:
                # Found multi-line command without backslash
                # Reconstruct with backslashes
                fixed_lines.append(line.rstrip() + ' \\')
                for k, cont_line in enumerate(continuation_lines):
                    if k < len(continuation_lines) - 1:
                        fixed_lines.append('    ' + cont_line + ' \\')
                    else:
                        # Last line doesn't need backslash
                        fixed_lines.append('    ' + cont_line)
                i = j
                continue
        
        # Regular line, keep as is
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)


def process_markdown_file(file_path: Path, dry_run: bool = False) -> bool:
    """Process a single markdown file and fix bash code blocks."""
    content = file_path.read_text()
    
    # Pattern to find bash code blocks
    # Matches ```bash ... ``` blocks
    pattern = r'```bash\n(.*?)```'
    
    modified = False
    
    def replace_block(match):
        nonlocal modified
        bash_content = match.group(1)
        fixed_content = fix_bash_block(bash_content)
        
        if fixed_content != bash_content:
            modified = True
            if not dry_run:
                print(f"  Fixed bash block in {file_path.name}")
            return f'```bash\n{fixed_content}```'
        return match.group(0)
    
    new_content = re.sub(pattern, replace_block, content, flags=re.DOTALL)
    
    if modified:
        if dry_run:
            print(f"Would fix: {file_path}")
        else:
            file_path.write_text(new_content)
            print(f"Fixed: {file_path}")
        return True
    
    return False


def main():
    """Main function to process all markdown files in pages directory."""
    dry_run = '--dry-run' in sys.argv
    
    # Get the project root (script is in scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    pages_dir = project_root / 'pages'
    
    if not pages_dir.exists():
        print(f"Error: pages directory not found at {pages_dir}")
        sys.exit(1)
    
    print(f"Scanning markdown files in {pages_dir}")
    if dry_run:
        print("DRY RUN MODE - no files will be modified")
    print()
    
    # Find all markdown files
    md_files = list(pages_dir.rglob('*.md'))
    
    total_files = len(md_files)
    modified_files = 0
    
    for md_file in sorted(md_files):
        if process_markdown_file(md_file, dry_run):
            modified_files += 1
    
    print()
    print(f"Processed {total_files} files")
    print(f"{'Would modify' if dry_run else 'Modified'}: {modified_files} files")
    
    if dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
